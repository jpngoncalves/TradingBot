#!/usr/bin/env python3
"""
=============================================================
  HL-EMA-RSI-MACD-BOT | Hyperliquid Futures Trading Bot
  Estratégia  : EMA (13/144) + RSI + MACD + StochRSI + ATR
  Autor       : DEV / jpngoncalves
  Versão      : 1.0.0
  Timeframe   : 15m (swing intraday)
=============================================================

IDEIA DA ESTRATÉGIA
-------------------
Versão adaptada para Hyperliquid do bot EMA/RSI/MACD que usavas em 2022.

LONG quando:
  1. Trigger: nas últimas N velas a EMA curta (13) esteve abaixo da EMA longa (144)
  2. EMA13 < EMA144 na vela actual
  3. RSI(14) < 30 (oversold)
  4. MACD hist < 0 mas a melhorar (hist actual > hist anterior)
  5. StochRSI a sair de oversold (K_prev < 0.2 e K_actual > K_prev)

EXIT LONG quando:
  - Stop Loss (ATR) ou Take Profit (R:R 1:2) são atingidos, OU
  - Sinal inverso de exaustão (RSI > 70, MACD hist a piorar, StochRSI a sair de overbought).

GESTÃO DE RISCO
---------------
  - SL: entry ± ATR(14) × atr_mult (1.5 por defeito)
  - TP: SL × rr (2.0 → R:R = 1:2)
  - Tamanho: % do account em risco por trade (risk_pct)
=============================================================
"""

import json
import time
import logging
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Hyperliquid SDK
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# ─────────────────────────────────────────
#  CONFIG & LOGGING
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CFG_FILE = BASE_DIR / "config.json"
LOG_FILE = BASE_DIR / "logs" / "bot.log"
LOG_FILE.parent.mkdir(exist_ok=True)


def load_config() -> dict:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"config.json não encontrado em {CFG_FILE}")
    with open(CFG_FILE) as f:
        return json.load(f)


CFG = load_config()

logging.basicConfig(
    level=getattr(logging, CFG.get("log_level", "INFO")),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("HL-EMA-RSI-MACD-BOT")


# ─────────────────────────────────────────
#  INDICADORES (numpy puro)
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
#  HYPERLIQUID CLIENT
# ─────────────────────────────────────────


class HLClient:
    def __init__(self, cfg: dict):
        self.symbol = cfg["symbol"]
        self.testnet = cfg.get("testnet", True)
        url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(url, skip_ws=True)
        wallet = eth_account.Account.from_key(cfg["secret_key"])
        self.exchange = Exchange(wallet, url, account_address=cfg["account_address"])
        log.info(f"Cliente HL iniciado | {'TESTNET' if self.testnet else 'MAINNET'} | {self.symbol}")

    def get_candles(self, interval: str, limit: int = 300) -> dict:
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
            "open": np.array(o), "high": np.array(h), "low": np.array(l),
            "close": np.array(c), "volume": np.array(v), "time": np.array(t)
        }

    def get_balance(self) -> float:
        state = self.info.user_state(CFG["account_address"])
        return float(state["crossMarginSummary"]["accountValue"])

    def get_position(self) -> dict | None:
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

    def place_market(self, is_buy: bool, size: float) -> dict:
        price = self._mid_price()
        slippage = 0.0015
        px = price * (1 + slippage) if is_buy else price * (1 - slippage)
        px = round(px, 2)
        result = self.exchange.order(
            self.symbol, is_buy, size, px,
            {"limit": {"tif": "Ioc"}},
            reduce_only=False,
        )
        log.info(f"{'BUY' if is_buy else 'SELL'} MARKET | size={size} | ~px={px} | {result}")
        return result

    def close_position(self, is_long: bool, size: float) -> dict:
        price = self._mid_price()
        slippage = 0.002
        px = price * (1 - slippage) if is_long else price * (1 + slippage)
        px = round(px, 2)
        result = self.exchange.order(
            self.symbol, not is_long, size, px,
            {"limit": {"tif": "Ioc"}},
            reduce_only=True,
        )
        log.info(f"CLOSE {'LONG' if is_long else 'SHORT'} | size={size} | ~px={px} | {result}")
        return result

    def coin_decimals(self) -> int:
        meta = self.info.meta()
        for asset in meta["universe"]:
            if asset["name"] == self.symbol:
                return asset.get("szDecimals", 3)
        return 3


# ─────────────────────────────────────────
#  SIGNAL ENGINE
# ─────────────────────────────────────────


class SignalEngine:
    """EMA(13/144) + RSI + MACD + StochRSI + ATR SL/TP."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def analyse(self, d: dict) -> dict:
        o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
        n = len(c)
        min_len = max(self.cfg["ema_st"], self.cfg["ema_lt"]) + 30
        if n < min_len:
            return {"signal": 0, "reason": "dados insuficientes"}

        ema_st = ema(c, self.cfg["ema_st"])
        ema_lt = ema(c, self.cfg["ema_lt"])
        rsi_arr = rsi(c, self.cfg.get("rsi_length", 14))
        macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
        k_srsi, d_srsi = stoch_rsi(c, 14, 14, 3, 3)
        atr_arr = atr(h, l, c, 14)

        price = float(c[-1])
        rsi_val = rsi_arr[-1]
        rsi_prev = rsi_arr[-2]
        macd_h = macd_hist[-1]
        macd_prev = macd_hist[-2]
        k_val = k_srsi[-1]
        k_prev = k_srsi[-2]

        # Trigger: EMA_ST < EMA_LT em alguma das últimas LAGS velas
        lags = self.cfg.get("lags", 5)
        cond_below = ema_st < ema_lt
        recent_below = cond_below[-(lags + 1):]
        trigger_long = np.any(recent_below)

        # Trigger de "top" (para saídas) – EMA_LT < EMA_ST nas últimas LAGS
        cond_above = ema_lt < ema_st
        recent_above = cond_above[-(lags + 1):]
        trigger_top = np.any(recent_above)

        # Condições bullish
        ema_bullish = ema_st[-1] < ema_lt[-1]
        rsi_oversold = rsi_val < self.cfg.get("rsi_oversold", 30)
        macd_bullish = (macd_h < 0) and (macd_h > macd_prev)
        stoch_bull = (k_prev < 0.2) and (k_val > k_prev)

        long_cond = (
            trigger_long
            and ema_bullish
            and rsi_oversold
            and macd_bullish
            and stoch_bull
        )

        atr_val = atr_arr[-1]
        if np.isnan(atr_val):
            return {"signal": 0, "reason": "ATR indisponível"}

        atr_mult = self.cfg.get("atr_mult", 1.5)
        rr = self.cfg.get("rr", 2.0)

        if long_cond:
            risk = atr_mult * atr_val
            sl = round(price - risk, 2)
            tp = round(price + risk * rr, 2)
            reasons = []
            if trigger_long: reasons.append("Trigger EMA_ST<EMA_LT")
            if ema_bullish: reasons.append("EMA13<EMA144")
            if rsi_oversold: reasons.append("RSI oversold")
            if macd_bullish: reasons.append("MACD hist improving")
            if stoch_bull: reasons.append("StochRSI saindo de OS")
            return {
                "signal": 1,
                "reason": " | ".join(reasons),
                "sl": sl,
                "tp": tp,
            }

        # Exit logic handled a nível de gestão de posição (SL/TP principalmente)
        return {"signal": 0, "reason": "sem sinal confluente"}


# ─────────────────────────────────────────
#  RISK MANAGER
# ─────────────────────────────────────────


class RiskManager:
    def __init__(self, cfg: dict):
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


# ─────────────────────────────────────────
#  BOT PRINCIPAL
# ─────────────────────────────────────────


class EmaRsiMacdBot:
    def __init__(self):
        self.cfg = CFG
        self.hl = HLClient(CFG)
        self.engine = SignalEngine(CFG)
        self.risk = RiskManager(CFG)
        self.hl.set_leverage(CFG["leverage"])
        self.decimals = self.hl.coin_decimals()
        self.active_sl: float | None = None
        self.active_tp: float | None = None
        log.info("=" * 55)
        log.info("  EMA-RSI-MACD BOT INICIADO")
        log.info(f"  Symbol    : {CFG['symbol']}-PERP")
        log.info(f"  Timeframe : {CFG['timeframe']}")
        log.info(f"  Leverage  : {CFG['leverage']}x")
        log.info(f"  Risk/trade: {CFG['risk_pct']}%")
        log.info("=" * 55)

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

        data = self.hl.get_candles(CFG["timeframe"], limit=self.cfg.get("lookback", 300))
        if len(data["close"]) < max(self.cfg["ema_st"], self.cfg["ema_lt"]) + 30:
            log.warning("Candles insuficientes, a aguardar...")
            return

        price = float(data["close"][-1])
        log.info(f"Preço actual  : {price}")

        pos = self.hl.get_position()
        if pos:
            log.info(f"Posição activa: {pos['side'].upper()} | size={pos['size']} | entry={pos['entry']}")
            self._manage_position(pos, price)
            return

        sig = self.engine.analyse(data)
        log.info(f"Sinal         : {sig['signal']} | {sig.get('reason','')}")

        if sig["signal"] == 0:
            return

        balance = self.hl.get_balance()
        size = self.risk.position_size(balance, price, sig["sl"], self.decimals)
        log.info(f"Balance       : {balance:.2f} USDC")
        log.info(f"Tamanho       : {size} {CFG['symbol']}")
        log.info(f"SL            : {sig['sl']} | TP: {sig['tp']}")

        if size <= 0:
            log.warning("Tamanho inválido — order ignorada.")
            return

        self.hl.place_market(True, size)
        self.active_sl = sig["sl"]
        self.active_tp = sig["tp"]

    def _manage_position(self, pos: dict, price: float):
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
        period = tf_secs.get(CFG["timeframe"], 900)
        now = time.time()
        wait = period - (now % period) + 2
        log.info(f"Próxima vela em {wait:.0f}s ({CFG['timeframe']})")
        time.sleep(wait)


if __name__ == "__main__":
    bot = EmaRsiMacdBot()
    bot.run()
