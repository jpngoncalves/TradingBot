#!/usr/bin/env python3
"""
HL-MOMENTUM-BOT — Hyperliquid Futures Trend + Momentum Bot

- Futuros perp na Hyperliquid (BTC-PERP por defeito)
- Timeframe 15m (configurável via config.json)
- Edge:
    * Tendência institucional: EMA 21/55/233 + VWAP
    * Pullbacks com confluência: RSI(14), MACD(12,26,9), StochRSI(14,3,3)
    * Gestão de risco: ATR(14) para SL/TP e tamanho da posição por % do account
"""

import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(divide="ignore", invalid="ignore")

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

BASE_DIR = Path(__file__).parent
CFG_FILE = BASE_DIR / "config.json"
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MONITOR_INTERVAL = 10  # segundos entre prints do monitor


def load_config() -> Dict[str, Any]:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"config.json não encontrado em {CFG_FILE}")
    with open(CFG_FILE, encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()

logging.basicConfig(
    level=getattr(logging, CFG.get("log_level", "INFO")),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("HL-MOMENTUM-BOT")


# ─────────────────────────────
#  INDICADORES
# ─────────────────────────────

def ema(series: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    out = np.full(len(series), np.nan, dtype=float)
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series)
    up   = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up   = np.empty(len(series), dtype=float)
    roll_down = np.empty(len(series), dtype=float)
    roll_up[:period]   = 0
    roll_down[:period] = 0
    roll_up[period]   = np.mean(up[:period])
    roll_down[period] = np.mean(down[:period])
    for i in range(period + 1, len(series)):
        roll_up[i]   = (roll_up[i-1]   * (period-1) + up[i-1])   / period
        roll_down[i] = (roll_down[i-1] * (period-1) + down[i-1]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(roll_down == 0, np.inf, roll_up / roll_down)
        out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out


def macd(series: np.ndarray, fast=12, slow=26, signal=9):
    ml = ema(series, fast) - ema(series, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl


def stoch_rsi(series: np.ndarray, rsi_len=14, stoch_len=14, k=3, d=3):
    r = rsi(series, rsi_len)
    out_k = np.full(len(series), np.nan, dtype=float)
    out_d = np.full(len(series), np.nan, dtype=float)
    for i in range(stoch_len, len(series)):
        w = r[i - stoch_len + 1:i + 1]
        lo, hi = np.nanmin(w), np.nanmax(w)
        out_k[i] = 0.5 if hi == lo else (r[i] - lo) / (hi - lo)
        if i >= stoch_len + k - 1:
            out_d[i] = np.nanmean(out_k[i - k + 1:i + 1])
    return out_k, out_d


def atr(highs, lows, closes, period=14) -> np.ndarray:
    pc = np.roll(closes, 1); pc[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - pc), np.abs(lows - pc)))
    out = np.full(len(tr), np.nan, dtype=float)
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        out[i] = (out[i-1] * (period-1) + tr[i]) / period
    return out


def vwap(highs, lows, closes, volumes) -> np.ndarray:
    tp  = (highs + lows + closes) / 3
    cv  = np.cumsum(volumes)
    ctv = np.cumsum(tp * volumes)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cv == 0, np.nan, ctv / cv)


def compute_indicators(d: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todos os indicadores e devolve dicionário com valores actuais."""
    h, l, c, v = d["high"], d["low"], d["close"], d["volume"]

    ef  = ema(c, cfg["ema_fast"])
    es  = ema(c, cfg["ema_slow"])
    et  = ema(c, cfg["ema_trend"])
    r   = rsi(c, cfg.get("rsi_length", 14))
    _, _, mh = macd(c, 12, 26, 9)
    k_s, _   = stoch_rsi(c, 14, 14, 3, 3)
    a   = atr(h, l, c, 14)
    vw  = vwap(h, l, c, v)

    price = float(c[-1])
    return {
        "price":      price,
        "ema_fast":   float(ef[-1]),
        "ema_slow":   float(es[-1]),
        "ema_trend":  float(et[-1]),
        "rsi":        float(r[-1]),
        "rsi_prev":   float(r[-2]),
        "macd_hist":  float(mh[-1]),
        "macd_prev":  float(mh[-2]),
        "stoch_k":    float(k_s[-1]),
        "stoch_prev": float(k_s[-2]),
        "atr":        float(a[-1]),
        "vwap":       float(vw[-1]),
    }


def print_monitor(ind: Dict[str, Any], pos: Optional[Dict], cfg: Dict[str, Any], next_candle_in: float):
    """Imprime painel de monitorização no terminal."""
    p = ind["price"]
    vw = ind["vwap"]
    ef, es, et = ind["ema_fast"], ind["ema_slow"], ind["ema_trend"]
    r  = ind["rsi"]
    mh = ind["macd_hist"]
    k  = ind["stoch_k"]
    a  = ind["atr"]

    # Tendência
    bull = p > vw and ef > es > et
    bear = p < vw and ef < es < et
    trend_str = "⬆️ BULLISH" if bull else ("⬇️ BEARISH" if bear else "➡️ LATERAL")

    # Condições long
    rsi_ok   = "✅" if r > 45 and r > ind["rsi_prev"]   else "❌"
    macd_ok  = "✅" if mh > 0 and mh > ind["macd_prev"] else "❌"
    stoch_ok = "✅" if ind["stoch_prev"] < 0.2 and k > ind["stoch_prev"] else "❌"
    trend_ok = "✅" if bull else "❌"

    all_ok = all(x == "✅" for x in [rsi_ok, macd_ok, stoch_ok, trend_ok])
    signal_str = "⚡ SINAL LONG DETECTADO!" if all_ok else "  Aguardar sinal..."

    pos_str = "Nenhuma"
    if pos:
        pnl_dist = p - pos["entry"] if pos["side"] == "long" else pos["entry"] - p
        pos_str = f"{pos['side'].upper()} | entry={pos['entry']} | size={pos['size']} | PnL dist={pnl_dist:+.2f}"

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"""
┌───────────────────────────────────────────────────────────┐
│  HL-MOMENTUM-BOT  [{ts} UTC]  {cfg['symbol']}-PERP {cfg['timeframe']}  │
├───────────────────────────────────────────────────────────┤
│  💰 Preço       : {p:<12.4f}   VWAP  : {vw:.4f}
│  📈 Tendência    : {trend_str}
│     EMA{cfg['ema_fast']:>3}    : {ef:<12.4f}   EMA{cfg['ema_slow']:>3}  : {es:.4f}
│     EMA{cfg['ema_trend']:>3}   : {et:.4f}
├───────────────────────────────────────────────────────────┤
│  CONDIÇÕES LONG
│  {trend_ok} Tendência bullish    (preço > VWAP + EMAs alinhadas)
│  {rsi_ok} RSI          : {r:.1f}  (prev: {ind['rsi_prev']:.1f})   [precisa >45 e a subir]
│  {macd_ok} MACD hist    : {mh:.4f}  (prev: {ind['macd_prev']:.4f})  [precisa >0 e a subir]
│  {stoch_ok} StochRSI K  : {k:.3f}  (prev: {ind['stoch_prev']:.3f})   [precisa sair <0.2]
│  📊 ATR          : {a:.4f}
├───────────────────────────────────────────────────────────┤
│  📦 Posição      : {pos_str}
│  {signal_str}
│  ⏱️  Próxima vela em: {next_candle_in:.0f}s
└───────────────────────────────────────────────────────────┘""", flush=True)


# ─────────────────────────────
#  HYPERLIQUID CLIENT
# ─────────────────────────────

class HLClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.symbol  = cfg["symbol"]
        self.testnet = cfg.get("testnet", True)
        url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(url, skip_ws=True)
        wallet = eth_account.Account.from_key(cfg["secret_key"])
        self.exchange = Exchange(wallet, url, account_address=cfg["account_address"])
        log.info(f"HLClient iniciado | {'TESTNET' if self.testnet else 'MAINNET'} | {self.symbol}")

    def get_candles(self, interval: str, limit: int = 500) -> Dict[str, np.ndarray]:
        now_ms = int(time.time() * 1000)
        tf_secs = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
                   "1h": 3600, "4h": 14400, "1d": 86400}
        since = now_ms - limit * tf_secs.get(interval, 900) * 1000
        req = {"type": "candleSnapshot", "req": {
            "coin": self.symbol, "interval": interval,
            "startTime": since, "endTime": now_ms,
        }}
        raw = self.info.post("/info", req)
        keys = ["o", "h", "l", "c", "v", "t"]
        arrs = {k: [] for k in keys}
        for candle in raw:
            for k in keys:
                arrs[k].append(float(candle[k]) if k != "t" else int(candle[k]))
        return {"open": np.array(arrs["o"], dtype=float),
                "high": np.array(arrs["h"], dtype=float),
                "low":  np.array(arrs["l"], dtype=float),
                "close":np.array(arrs["c"], dtype=float),
                "volume":np.array(arrs["v"], dtype=float),
                "time": np.array(arrs["t"])}

    def get_balance(self) -> float:
        return float(self.info.user_state(CFG["account_address"])["crossMarginSummary"]["accountValue"])

    def get_position(self) -> Optional[Dict[str, Any]]:
        for pos in self.info.user_state(CFG["account_address"]).get("assetPositions", []):
            if pos["position"]["coin"] == self.symbol:
                size = float(pos["position"]["szi"])
                if size != 0:
                    return {"side": "long" if size > 0 else "short",
                            "size": abs(size),
                            "entry": float(pos["position"]["entryPx"])}
        return None

    def set_leverage(self, lev: int):
        try:
            self.exchange.update_leverage(lev, self.symbol, is_cross=True)
            log.info(f"Leverage definida: {lev}x")
        except Exception as e:
            log.warning(f"Erro ao definir leverage: {e}")

    def _mid_price(self) -> float:
        return float(self.info.all_mids()[self.symbol])

    def place_market(self, is_buy: bool, size: float):
        price = self._mid_price()
        px = round(price * (1.0015 if is_buy else 0.9985), 2)
        result = self.exchange.order(self.symbol, is_buy, size, px,
                                     {"limit": {"tif": "Ioc"}}, reduce_only=False)
        log.info(f"{'BUY' if is_buy else 'SELL'} MARKET | size={size} | ~px={px} | {result}")
        return result

    def close_position(self, is_long: bool, size: float):
        price = self._mid_price()
        px = round(price * (0.998 if is_long else 1.002), 2)
        result = self.exchange.order(self.symbol, not is_long, size, px,
                                     {"limit": {"tif": "Ioc"}}, reduce_only=True)
        log.info(f"CLOSE {'LONG' if is_long else 'SHORT'} | size={size} | ~px={px} | {result}")
        return result

    def coin_decimals(self) -> int:
        for asset in self.info.meta()["universe"]:
            if asset["name"] == self.symbol:
                return asset.get("szDecimals", 3)
        return 3


class RiskManager:
    def __init__(self, cfg):
        self.risk_pct = cfg["risk_pct"] / 100

    def position_size(self, balance, entry, sl, decimals=3) -> float:
        dist = abs(entry - sl) / entry
        if dist == 0:
            return 0.0
        size = (balance * self.risk_pct / dist) / entry
        f = 10 ** decimals
        return max(math.floor(size * f) / f, 10 ** (-decimals))


class SignalEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def analyse(self, d) -> Dict[str, Any]:
        c = d["close"]
        min_len = max(self.cfg["ema_fast"], self.cfg["ema_slow"], self.cfg["ema_trend"]) + 50
        if len(c) < min_len:
            return {"signal": 0, "reason": f"dados insuficientes ({len(c)}/{min_len} candles)"}

        ind = compute_indicators(d, self.cfg)

        if any(math.isnan(ind[k]) for k in ["rsi", "macd_hist", "stoch_k", "atr", "vwap"]):
            return {"signal": 0, "reason": "indicadores incompletos", "ind": ind}

        bull = ind["price"] > ind["vwap"] and ind["ema_fast"] > ind["ema_slow"] > ind["ema_trend"]
        atr_mult = self.cfg.get("atr_mult", 1.5)
        rr       = self.cfg.get("rr", 2.0)

        long_ok = (bull
                   and ind["rsi"] > 45 and ind["rsi"] > ind["rsi_prev"]
                   and ind["macd_hist"] > 0 and ind["macd_hist"] > ind["macd_prev"]
                   and ind["stoch_prev"] < 0.2 and ind["stoch_k"] > ind["stoch_prev"])
        if long_ok:
            risk = atr_mult * ind["atr"]
            return {"signal": 1, "side": "long",
                    "sl": round(ind["price"] - risk, 2),
                    "tp": round(ind["price"] + risk * rr, 2),
                    "reason": "trend bullish | RSI OK | MACD OK | StochRSI OS->UP",
                    "ind": ind}

        if self.cfg.get("enable_shorts", False):
            bear = ind["price"] < ind["vwap"] and ind["ema_fast"] < ind["ema_slow"] < ind["ema_trend"]
            short_ok = (bear
                        and ind["rsi"] < 55 and ind["rsi"] < ind["rsi_prev"]
                        and ind["macd_hist"] < 0 and ind["macd_hist"] < ind["macd_prev"]
                        and ind["stoch_prev"] > 0.8 and ind["stoch_k"] < ind["stoch_prev"])
            if short_ok:
                risk = atr_mult * ind["atr"]
                return {"signal": 1, "side": "short",
                        "sl": round(ind["price"] + risk, 2),
                        "tp": round(ind["price"] - risk * rr, 2),
                        "reason": "trend bearish | RSI OK | MACD OK | StochRSI OB->DOWN",
                        "ind": ind}

        return {"signal": 0, "reason": "sem confluência", "ind": ind}


class MomentumBot:
    def __init__(self):
        self.cfg      = CFG
        self.hl       = HLClient(self.cfg)
        self.engine   = SignalEngine(self.cfg)
        self.risk     = RiskManager(self.cfg)
        self.hl.set_leverage(self.cfg["leverage"])
        self.decimals = self.hl.coin_decimals()
        self.active_sl: Optional[float] = None
        self.active_tp: Optional[float] = None
        log.info("=" * 70)
        log.info("  HL-MOMENTUM-BOT INICIADO")
        log.info(f"  Symbol     : {self.cfg['symbol']}-PERP")
        log.info(f"  Timeframe  : {self.cfg['timeframe']}")
        log.info(f"  Leverage   : {self.cfg['leverage']}x")
        log.info(f"  Risk/trade : {self.cfg['risk_pct']}%")
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

    def _tick(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"── Tick {ts} ─────────────────────────")

        data  = self.hl.get_candles(self.cfg["timeframe"], self.cfg.get("lookback", 500))
        price = float(data["close"][-1])
        log.info(f"Preço actual  : {price}")

        pos = self.hl.get_position()
        if pos:
            log.info(f"Posição activa: {pos['side'].upper()} | size={pos['size']} | entry={pos['entry']}")
            self._manage_position(pos, price)
            self._monitor_loop(data, pos)
            return

        sig = self.engine.analyse(data)
        log.info(f"Sinal         : {sig['signal']} | {sig.get('side','-')} | {sig.get('reason','')}")

        if sig["signal"] == 1:
            balance = self.hl.get_balance()
            size    = self.risk.position_size(balance, price, sig["sl"], self.decimals)
            log.info(f"Balance: {balance:.2f} USDC | Tamanho: {size} | SL: {sig['sl']} | TP: {sig['tp']}")
            if size > 0:
                self.hl.place_market(sig["side"] == "long", size)
                self.active_sl = sig["sl"]
                self.active_tp = sig["tp"]
            else:
                log.warning("Tamanho inválido — order ignorada.")

        # — Monitor loop até próxima vela —
        self._monitor_loop(data, pos)

    def _monitor_loop(self, data_snapshot, pos_snapshot):
        """Imprime o painel de monitorização a cada MONITOR_INTERVAL segundos
        até à próxima vela."""
        tf_secs = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
                   "1h": 3600, "4h": 14400, "1d": 86400}
        period  = tf_secs.get(self.cfg["timeframe"], 900)
        now     = time.time()
        candle_end = now + (period - (now % period)) + 2

        # Pre-calcula indicadores do snapshot actual
        min_len = max(self.cfg["ema_fast"], self.cfg["ema_slow"], self.cfg["ema_trend"]) + 50
        has_ind = len(data_snapshot["close"]) >= min_len

        while True:
            remaining = candle_end - time.time()
            if remaining <= 0:
                break

            # Actualiza preço em tempo real
            try:
                live_price = float(self.hl.info.all_mids()[self.cfg["symbol"]])
                # Actualiza último close no snapshot para indicadores
                data_snapshot["close"][-1] = live_price
            except Exception:
                live_price = float(data_snapshot["close"][-1])

            if has_ind:
                try:
                    ind = compute_indicators(data_snapshot, self.cfg)
                    # Actualiza posição
                    pos = self.hl.get_position()
                    print_monitor(ind, pos, self.cfg, remaining)
                except Exception as e:
                    log.debug(f"Monitor error: {e}")
            else:
                print(f"\r  💰 {self.cfg['symbol']} = {live_price}  |  A carregar indicadores...  "
                      f"Próxima vela em {remaining:.0f}s", end="", flush=True)

            time.sleep(MONITOR_INTERVAL)

    def _manage_position(self, pos, price):
        if self.active_sl is None or self.active_tp is None:
            log.warning("SL/TP não definidos para posição activa")
            return
        is_long = pos["side"] == "long"
        if (is_long and price >= self.active_tp) or (not is_long and price <= self.active_tp):
            log.info(f"TAKE PROFIT @ {price} (TP={self.active_tp})")
            self.hl.close_position(is_long, pos["size"])
            self._reset()
        elif (is_long and price <= self.active_sl) or (not is_long and price >= self.active_sl):
            log.info(f"STOP LOSS @ {price} (SL={self.active_sl})")
            self.hl.close_position(is_long, pos["size"])
            self._reset()
        else:
            log.info(f"Posição OK → dist_TP={abs(price-self.active_tp):.2f} | dist_SL={abs(price-self.active_sl):.2f}")

    def _reset(self):
        self.active_sl = None
        self.active_tp = None


if __name__ == "__main__":
    bot = MomentumBot()
    bot.run()
