#!/usr/bin/env python3
"""
HL-LIQ-MAP-BOT — Hyperliquid Futures + Coinalyze Liquidation-Aware Bot

Versão actualizada para usar a API gratuita da Coinalyze em vez da CoinGlass.

- Futuros perp na Hyperliquid (BTC-PERP por defeito)
- Timeframe 15m (configurável)
- Edge:
  1) Tendência institucional (EMA21/55/233 + VWAP)
  2) Pullbacks de qualidade (RSI, MACD, StochRSI)
  3) Filtro de liquidações da Coinalyze (liquidation history) para evitar
     entrar contra clusters grandes de liquidações recentes e alinhar com
     assimetrias long/short.

Coinalyze API docs: https://api.coinalyze.net/v1/doc/
Endpoint usado: GET /liquidation_history
  - params: symbols, interval, from, to, convert_to_usd
  - resposta: [ { symbol, history: [ { t, l, s } ] } ]
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import requests

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

BASE_DIR = Path(__file__).parent
CFG_FILE = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
  if not CFG_FILE.exists():
    raise FileNotFoundError(f"config.json não encontrado em {CFG_FILE}")
  with open(CFG_FILE, encoding="utf-8") as f:
    cfg = json.load(f)
  # permitir override via env var
  api_key_env = os.getenv("COINALYZE_API_KEY")
  if api_key_env:
    cfg["coinalyze_api_key"] = api_key_env
  return cfg


CFG = load_config()

logging.basicConfig(
  level=getattr(logging, CFG.get("log_level", "INFO")),
  format="%(asctime)s | %(levelname)-8s | %(message)s",
  handlers=[
    logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
    logging.StreamHandler(sys.stdout),
  ],
)
log = logging.getLogger("HL-LIQ-MAP-BOT")


# ─────────────────────────────
#  INDICADORES
# ─────────────────────────────


def ema(series: np.ndarray, period: int) -> np.ndarray:
  k = 2 / (period + 1)
  out = np.full_like(series, np.nan)
  out[period - 1] = np.mean(series[:period])
  for i in range(period, len(series)):
    out[i] = series[i] * k + out[i - 1] * (1 - k)
  return out


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
  delta = np.diff(series)
  up = np.where(delta > 0, delta, 0.0)
  down = np.where(delta < 0, -delta, 0.0)
  roll_up = np.empty_like(series)
  roll_down = np.empty_like(series)
  roll_up[:period] = 0
  roll_down[:period] = 0
  roll_up[period] = np.mean(up[:period])
  roll_down[period] = np.mean(down[:period])
  for i in range(period + 1, len(series)):
    roll_up[i] = (roll_up[i-1] * (period - 1) + up[i-1]) / period
    roll_down[i] = (roll_down[i-1] * (period - 1) + down[i-1]) / period
  rs = np.where(roll_down == 0, np.inf, roll_up / roll_down)
  rsi_val = 100 - (100 / (1 + rs))
  rsi_val[:period] = np.nan
  return rsi_val


def macd(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
  ema_fast = ema(series, fast)
  ema_slow = ema(series, slow)
  macd_line = ema_fast - ema_slow
  signal_line = ema(macd_line, signal)
  hist = macd_line - signal_line
  return macd_line, signal_line, hist


def stoch_rsi(series: np.ndarray, rsi_length: int = 14, stoch_length: int = 14,
              k: int = 3, d: int = 3):
  r = rsi(series, rsi_length)
  out_k = np.full_like(series, np.nan, dtype=float)
  out_d = np.full_like(series, np.nan, dtype=float)
  for i in range(stoch_length, len(series)):
    window = r[i - stoch_length + 1:i + 1]
    r_min = np.nanmin(window)
    r_max = np.nanmax(window)
    if r_max - r_min == 0:
      k_val = 0.5
    else:
      k_val = (r[i] - r_min) / (r_max - r_min)
    out_k[i] = k_val
    if i >= stoch_length + k - 1:
      out_d[i] = np.nanmean(out_k[i - k + 1:i + 1])
  return out_k, out_d


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
  prev_close = np.roll(closes, 1)
  prev_close[0] = closes[0]
  tr1 = highs - lows
  tr2 = np.abs(highs - prev_close)
  tr3 = np.abs(lows - prev_close)
  tr = np.maximum(tr1, np.maximum(tr2, tr3))
  out = np.full_like(tr, np.nan, dtype=float)
  out[period - 1] = np.mean(tr[:period])
  for i in range(period, len(tr)):
    out[i] = (out[i-1] * (period - 1) + tr[i]) / period
  return out


def vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
  typical_price = (highs + lows + closes) / 3
  cum_vol = np.cumsum(volumes)
  cum_tp_vol = np.cumsum(typical_price * volumes)
  vwap_arr = np.where(cum_vol == 0, np.nan, cum_tp_vol / cum_vol)
  return vwap_arr


# ─────────────────────────────
#  HYPERLIQUID CLIENT
# ─────────────────────────────


class HLClient:
  def __init__(self, cfg: Dict[str, Any]):
    self.symbol = cfg["symbol"]
    self.testnet = cfg.get("testnet", True)
    url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
    self.info = Info(url, skip_ws=True)
    wallet = eth_account.Account.from_key(cfg["secret_key"])
    self.exchange = Exchange(wallet, url, account_address=cfg["account_address"])
    log.info(f"HLClient iniciado | {'TESTNET' if self.testnet else 'MAINNET'} | {self.symbol}")

  def get_candles(self, interval: str, limit: int = 300) -> Dict[str, np.ndarray]:
    now_ms = int(time.time() * 1000)
    tf_secs = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
               "1h": 3600, "4h": 14400, "1d": 86400}
    tf_ms = tf_secs.get(interval, 900) * 1000
    since = now_ms - limit * tf_ms
    req = {"type": "candleSnapshot", "req": {
      "coin": self.symbol,
      "interval": interval,
      "startTime": since,
      "endTime": now_ms,
    }}
    raw = self.info.post("/info", req)
    o, h, l, c, v, t = [], [], [], [], [], []
    for candle in raw:
      o.append(float(candle["o"]))
      h.append(float(candle["h"]))
      l.append(float(candle["l"]))
      c.append(float(candle["c"]))
      v.append(float(candle["v"]))
      t.append(int(candle["t"]))
    return {
      "open": np.array(o),
      "high": np.array(h),
      "low": np.array(l),
      "close": np.array(c),
      "volume": np.array(v),
      "time": np.array(t),
    }

  def get_balance(self) -> float:
    state = self.info.user_state(CFG["account_address"])
    return float(state["crossMarginSummary"]["accountValue"])

  def get_position(self) -> Optional[Dict[str, Any]]:
    state = self.info.user_state(CFG["account_address"])
    for pos in state.get("assetPositions", []):
      if pos["position"]["coin"] == self.symbol:
        size = float(pos["position"]["szi"])
        if size != 0:
          return {
            "side": "long" if size > 0 else "short",
            "size": abs(size),
            "entry": float(pos["position"]["entryPx"]),
          }
    return None

  def set_leverage(self, lev: int):
    try:
      self.exchange.update_leverage(lev, self.symbol, is_cross=True)
      log.info(f"Leverage definida: {lev}x")
    except Exception as e:
      log.warning(f"Erro ao definir leverage: {e}")

  def _mid_price(self) -> float:
    mids = self.info.all_mids()
    return float(mids[self.symbol])

  def place_market(self, is_buy: bool, size: float) -> Dict[str, Any]:
    price = self._mid_price()
    slippage = 0.0015
    px = price * (1 + slippage) if is_buy else price * (1 - slippage)
    px = round(px, 2)
    result = self.exchange.order(
      self.symbol,
      is_buy,
      size,
      px,
      {"limit": {"tif": "Ioc"}},
      reduce_only=False,
    )
    log.info(f"{'BUY' if is_buy else 'SELL'} MARKET | size={size} | ~px={px} | {result}")
    return result

  def close_position(self, is_long: bool, size: float) -> Dict[str, Any]:
    price = self._mid_price()
    slippage = 0.002
    px = price * (1 - slippage) if is_long else price * (1 + slippage)
    px = round(px, 2)
    result = self.exchange.order(
      self.symbol,
      not is_long,
      size,
      px,
      {"limit": {"tif": "Ioc"}},
      reduce_only=True,
    )
    log.info(f"CLOSE {'LONG' if is_long else 'SHORT'} | size={size} | ~px={px}")
    return result

  def coin_decimals(self) -> int:
    meta = self.info.meta()
    for asset in meta["universe"]:
      if asset["name"] == self.symbol:
        return asset.get("szDecimals", 3)
    return 3


# ─────────────────────────────
#  COINALYZE LIQUIDATION CLIENT
# ─────────────────────────────


@dataclass
class LiqPoint:
  t: int  # timestamp (s)
  l: float  # long liquidations
  s: float  # short liquidations


class CoinalyzeClient:
  """Cliente mínimo para o endpoint liquidation history.

  Docs oficiais: https://api.coinalyze.net/v1/doc/

  GET /liquidation_history
    query:
      - api_key
      - symbols
      - interval
      - from, to (UNIX seconds)
      - convert_to_usd
  Response:
    [
      {
        "symbol": "BTCUSDT_PERP.A",
        "history": [ { "t": 0, "l": 0, "s": 0 }, ... ]
      }
    ]
  """

  def __init__(self, cfg: Dict[str, Any]):
    self.api_key = cfg.get("coinalyze_api_key")
    self.base_url = cfg.get("coinalyze_base_url", "https://api.coinalyze.net/v1")
    self.symbol = cfg.get("coinalyze_symbol", "BTCUSDT_PERP.A")

  def enabled(self) -> bool:
    return bool(self.api_key)

  def _params(self, extra: Dict[str, Any]) -> Dict[str, Any]:
    p = {"api_key": self.api_key}
    p.update(extra)
    return p

  def fetch_recent_liq(self, interval: str = "15min", lookback_intervals: int = 96) -> Optional[list]:
    """Vai buscar ~lookback_intervals de histórico de liquidações.

    - interval: 1min, 5min, 15min, ...
    - lookback_intervals: quantas velas pretendemos (<= 1500/2000 por limite Coinalyze)
    """
    if not self.enabled():
      log.warning("Coinalyze API key não configurada - módulo de liq off")
      return None

    now = int(time.time())
    # margem de segurança: lookback_intervals * interval_sec
    step = {
      "1min": 60,
      "5min": 300,
      "15min": 900,
      "30min": 1800,
      "1hour": 3600,
    }.get(interval, 900)
    total = step * lookback_intervals
    frm = now - total
    to = now

    url = f"{self.base_url}/liquidation_history"
    params = self._params({
      "symbols": self.symbol,
      "interval": interval,
      "from": frm,
      "to": to,
      "convert_to_usd": "true",
    })

    try:
      resp = requests.get(url, params=params, timeout=5)
      if resp.status_code != 200:
        log.warning(f"Coinalyze HTTP {resp.status_code}: {resp.text[:200]}")
        return None
      data = resp.json()
      if not data:
        return None
      return data[0].get("history", [])
    except Exception as e:
      log.warning(f"Erro Coinalyze: {e}")
      return None

  def analyse_liq(self, price: float, liq_history: Optional[list], window_hours: float = 24.0) -> Dict[str, Any]:
    """Analisa liquidações recentes perto do preço actual.

    Aqui não temos price por ponto (só l/s por candle), portanto usamos
    assimetria temporal:
      - somamos liquidações long/short nas últimas X horas
      - interpretamos net_short (s - l) e net_long (l - s) como sinal de
        posição forçada predominante.
    """
    if not liq_history:
      return {"enabled": False, "reason": "sem dados Coinalyze"}

    now = int(time.time())
    cutoff = now - int(window_hours * 3600)

    long_recent = 0.0
    short_recent = 0.0
    for p in liq_history:
      t = int(p.get("t", 0))
      if t < cutoff:
        continue
      l = float(p.get("l", 0.0))
      s = float(p.get("s", 0.0))
      long_recent += l
      short_recent += s

    net_short = short_recent - long_recent
    net_long = long_recent - short_recent

    return {
      "enabled": True,
      "long_recent": long_recent,
      "short_recent": short_recent,
      "net_short": net_short,
      "net_long": net_long,
    }


# ─────────────────────────────
#  RISK MANAGER
# ─────────────────────────────


class RiskManager:
  def __init__(self, cfg: Dict[str, Any]):
    self.risk_pct = cfg["risk_pct"] / 100
    self.leverage = cfg["leverage"]

  def position_size(self, balance: float, entry: float, sl: float, decimals: int = 3) -> float:
    risk_usd = balance * self.risk_pct
    dist_pct = abs(entry - sl) / entry
    if dist_pct == 0:
      return 0.0
    size = (risk_usd / dist_pct) / entry
    factor = 10 ** decimals
    size = math.floor(size * factor) / factor
    return max(size, 10 ** (-decimals))


# ─────────────────────────────
#  SIGNAL ENGINE
# ─────────────────────────────


class SignalEngine:
  def __init__(self, cfg: Dict[str, Any]):
    self.cfg = cfg

  def analyse(self, d: Dict[str, np.ndarray], liq_info: Dict[str, Any]) -> Dict[str, Any]:
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    n = len(c)
    min_len = max(self.cfg["ema_fast"], self.cfg["ema_slow"], self.cfg["ema_trend"]) + 50
    if n < min_len:
      return {"signal": 0, "reason": "dados insuficientes"}

    ema_fast = ema(c, self.cfg["ema_fast"])
    ema_slow = ema(c, self.cfg["ema_slow"])
    ema_trend = ema(c, self.cfg["ema_trend"])
    rsi_arr = rsi(c, self.cfg.get("rsi_length", 14))
    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    k_srsi, d_srsi = stoch_rsi(c, 14, 14, 3, 3)
    atr_arr = atr(h, l, c, 14)
    vwap_arr = vwap(h, l, c, v)

    price = float(c[-1])
    rsi_val = rsi_arr[-1]
    rsi_prev = rsi_arr[-2]
    macd_h = macd_hist[-1]
    macd_prev = macd_hist[-2]
    k_val = k_srsi[-1]
    k_prev = k_srsi[-2]
    atr_val = atr_arr[-1]
    vwap_val = vwap_arr[-1]

    if np.isnan([rsi_val, macd_h, k_val, atr_val, vwap_val]).any():
      return {"signal": 0, "reason": "indicadores incompletos"}

    bull_trend = (price > vwap_val) and (ema_fast[-1] > ema_slow[-1] > ema_trend[-1])
    bear_trend = (price < vwap_val) and (ema_fast[-1] < ema_slow[-1] < ema_trend[-1])

    rsi_pullback_long = (rsi_val > 45) and (rsi_val > rsi_prev)
    rsi_pullback_short = (rsi_val < 55) and (rsi_val < rsi_prev)

    macd_long = (macd_h > 0) and (macd_h > macd_prev)
    macd_short = (macd_h < 0) and (macd_h < macd_prev)

    stoch_long = (k_prev < 0.2) and (k_val > k_prev)
    stoch_short = (k_prev > 0.8) and (k_val < k_prev)

    liq_enabled = liq_info.get("enabled", False)
    long_ok = short_ok = True
    liq_reason = ""
    if liq_enabled:
      net_short = liq_info.get("net_short", 0.0)
      net_long = liq_info.get("net_long", 0.0)
      # LONG: queremos mais shorts liquidados recentemente → net_short>0
      long_ok = net_short > 0
      # SHORT: mais longs liquidados recentemente → net_long>0
      short_ok = net_long > 0
      liq_reason = f"net_short={net_short:.0f}, net_long={net_long:.0f}"

    long_cond = bull_trend and rsi_pullback_long and macd_long and stoch_long and long_ok
    short_cond = bear_trend and rsi_pullback_short and macd_short and stoch_short and short_ok

    atr_mult = self.cfg.get("atr_mult", 1.5)
    rr = self.cfg.get("rr", 2.0)

    if long_cond:
      risk = atr_mult * atr_val
      sl = round(price - risk, 2)
      tp = round(price + risk * rr, 2)
      reasons = [
        "trend bullish (EMA/VWAP)",
        "RSI pullback up", "MACD>0 improving", "StochRSI OS->UP",
      ]
      if liq_reason:
        reasons.append(f"Coinalyze: {liq_reason}")
      return {
        "signal": 1,
        "side": "long",
        "sl": sl,
        "tp": tp,
        "reason": " | ".join(reasons),
      }

    if short_cond and self.cfg.get("enable_shorts", False):
      risk = atr_mult * atr_val
      sl = round(price + risk, 2)
      tp = round(price - risk * rr, 2)
      reasons = [
        "trend bearish (EMA/VWAP)",
        "RSI pullback down", "MACD<0 worsening", "StochRSI OB->DOWN",
      ]
      if liq_reason:
        reasons.append(f"Coinalyze: {liq_reason}")
      return {
        "signal": 1,
        "side": "short",
        "sl": sl,
        "tp": tp,
        "reason": " | ".join(reasons),
      }

    return {"signal": 0, "reason": f"sem confluência | {liq_reason}"}


# ─────────────────────────────
#  BOT PRINCIPAL
# ─────────────────────────────


class LiquidityMapBot:
  def __init__(self):
    self.cfg = CFG
    self.hl = HLClient(self.cfg)
    self.coinalyze = CoinalyzeClient(self.cfg)
    self.engine = SignalEngine(self.cfg)
    self.risk = RiskManager(self.cfg)
    self.hl.set_leverage(self.cfg["leverage"])
    self.decimals = self.hl.coin_decimals()
    self.active_sl: Optional[float] = None
    self.active_tp: Optional[float] = None
    log.info("=" * 70)
    log.info("  HL-LIQ-MAP-BOT (Coinalyze) INICIADO")
    log.info(f"  Symbol     : {self.cfg['symbol']}-PERP")
    log.info(f"  Timeframe  : {self.cfg['timeframe']}")
    log.info(f"  Leverage   : {self.cfg['leverage']}x")
    log.info(f"  Risk/trade : {self.cfg['risk_pct']}%")
    log.info(f"  Coinalyze  : {'ON' if self.coinalyze.enabled() else 'OFF'}")
    log.info("=" * 70)

  def run(self):
    log.info("Bot a correr... Ctrl+C para parar.")
    while True:
      try:
        self._tick()
      except KeyboardInterrupt:
        log.info("Bot parado pelo utilizador.")
        break
      except Exception as e:
        log.error(f"Erro no loop principal: {e}", exc_info=True)
        time.sleep(30)
      self._sleep_until_next_candle()

  def _tick(self):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"── Tick {ts} ──────────────────────────")

    data = self.hl.get_candles(self.cfg["timeframe"], limit=self.cfg.get("lookback", 400))
    price = float(data["close"][-1])
    log.info(f"Preço actual  : {price}")

    liq_info = {"enabled": False}
    if self.coinalyze.enabled():
      liq_hist = self.coinalyze.fetch_recent_liq(interval="15min", lookback_intervals=96)
      liq_info = self.coinalyze.analyse_liq(price, liq_hist, window_hours=24.0)
      log.info(f"Coinalyze liq: {liq_info}")

    pos = self.hl.get_position()
    if pos:
      log.info(f"Posição activa: {pos['side'].upper()} | size={pos['size']} | entry={pos['entry']}")
      self._manage_position(pos, price)
      return

    sig = self.engine.analyse(data, liq_info)
    log.info(f"Sinal         : {sig['signal']} | {sig.get('side','-')} | {sig.get('reason','')}")

    if sig["signal"] == 0:
      return

    balance = self.hl.get_balance()
    size = self.risk.position_size(balance, price, sig["sl"], self.decimals)
    log.info(f"Balance       : {balance:.2f} USDC")
    log.info(f"Tamanho       : {size} {self.cfg['symbol']}")
    log.info(f"SL            : {sig['sl']} | TP: {sig['tp']}")

    if size <= 0:
      log.warning("Tamanho inválido — order ignorada.")
      return

    is_long = sig["side"] == "long"
    self.hl.place_market(is_long, size)
    self.active_sl = sig["sl"]
    self.active_tp = sig["tp"]

  def _manage_position(self, pos: Dict[str, Any], price: float):
    if self.active_sl is None or self.active_tp is None:
      log.warning("SL/TP não definidos para posição activa")
      return

    is_long = pos["side"] == "long"
    hit_sl = price <= self.active_sl if is_long else price >= self.active_sl
    hit_tp = price >= self.active_tp if is_long else price <= self.active_tp

    if hit_tp:
      log.info(f"TAKE PROFIT atingido @ {price} (TP={self.active_tp})")
      self.hl.close_position(is_long, pos["size"])
      self._reset_state()
    elif hit_sl:
      log.info(f"STOP LOSS atingido @ {price} (SL={self.active_sl})")
      self.hl.close_position(is_long, pos["size"])
      self._reset_state()
    else:
      dist_tp = abs(price - self.active_tp)
      dist_sl = abs(price - self.active_sl)
      log.info(f"Posição activa → dist_TP={dist_tp:.2f} | dist_SL={dist_sl:.2f}")

  def _reset_state(self):
    self.active_sl = None
    self.active_tp = None

  def _sleep_until_next_candle(self):
    tf_secs = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
               "1h": 3600, "4h": 14400, "1d": 86400}
    period = tf_secs.get(self.cfg["timeframe"], 900)
    now = time.time()
    wait = period - (now % period) + 2
    log.info(f"Próxima vela em {wait:.0f}s ({self.cfg['timeframe']})")
    time.sleep(wait)


if __name__ == "__main__":
  bot = LiquidityMapBot()
  bot.run()
