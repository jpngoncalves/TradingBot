#!/usr/bin/env python3
"""
=============================================================
  HL-STOCH-RSI-MACD-BOT | Hyperliquid Futures Trading Bot
  Estratégia  : Stoch Slow + RSI + MACD + Trend (EMA/VWAP)
  Autor       : DEV / jpngoncalves
  Versão      : 1.0.0
  Timeframe   : 5m (scalping / intraday)
=============================================================

IDEIA DA ESTRATÉGIA
-------------------
Baseada no vídeo "Trading bot - Stochastics Slow, RSI, MACD" adaptado para
Hyperliquid Futures:

LONG quando:
  1. EMA8 > EMA21 > EMA55 (tendência bullish)
  2. Preço acima do VWAP
  3. Stoch Slow (%K e %D) esteve em oversold (<20) nas últimas N velas
  4. Agora %K e %D estão entre 20 e 80 (pullback concluído)
  5. RSI(14) > 50
  6. MACD hist > 0

SHORT quando (simétrico):
  1. EMA8 < EMA21 < EMA55 (tendência bearish)
  2. Preço abaixo do VWAP
  3. Stoch Slow esteve em overbought (>80) nas últimas N velas
  4. Agora %K e %D estão entre 20 e 80
  5. RSI(14) < 50
  6. MACD hist < 0

GESTÃO DE RISCO
---------------
  - Stop Loss   : multiplicador fixo sobre o preço (ex.: 0.99 = -1%)
  - Take Profit : multiplicador fixo (ex.: 1.02 = +2%)
  - Tamanho     : % do account em risco por trade (config)
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
log = logging.getLogger("HL-STOCH-BOT")


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


def stoch_slow(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
    """Stoch Slow %K,%D em [0,100]."""
    n = len(closes)
    k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        window_h = highs[i - k_period + 1:i + 1]
        window_l = lows[i - k_period + 1:i + 1]
        h_max = np.max(window_h)
        l_min = np.min(window_l)
        if h_max - l_min == 0:
            k[i] = 50
        else:
            k[i] = (closes[i] - l_min) / (h_max - l_min) * 100
    # smooth %K
    k_smooth = np.full(n, np.nan)
    for i in range(smooth_k - 1, n):
        k_smooth[i] = np.nanmean(k[i - smooth_k + 1:i + 1])
    # %D
    d = np.full(n, np.nan)
    for i in range(d_period - 1, n):
        d[i] = np.nanmean(k_smooth[i - d_period + 1:i + 1])
    return k_smooth, d


def vwap(closes, highs, lows, volumes):
    hlc3 = (highs + lows + closes) / 3
    cum_vol = np.cumsum(volumes)
    cum_pv = np.cumsum(hlc3 * volumes)
    return cum_pv / np.where(cum_vol == 0, 1, cum_vol)


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
        tf_ms = tf_secs.get(interval, 300) * 1000
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
    """Stoch Slow + RSI + MACD + trend filter."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def analyse(self, d: dict) -> dict:
        o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
        n = len(c)
        min_len = 80
        if n < min_len:
            return {"signal": 0, "reason": "dados insuficientes"}

        e8 = ema(c, self.cfg["ema_fast"])
        e21 = ema(c, self.cfg["ema_slow"])
        e55 = ema(c, self.cfg["ema_trend"])
        vwap_arr = vwap(c, h, l, v)

        rsi_arr = rsi(c, self.cfg.get("rsi_length", 14))
        macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
        k, d_slow = stoch_slow(h, l, c,
                               k_period=self.cfg.get("stoch_k", 14),
                               d_period=self.cfg.get("stoch_d", 3),
                               smooth_k=self.cfg.get("stoch_smooth", 3))

        price = float(c[-1])
        ema_bull = e8[-1] > e21[-1] > e55[-1]
        ema_bear = e8[-1] < e21[-1] < e55[-1]

        above_vwap = price > vwap_arr[-1]
        below_vwap = price < vwap_arr[-1]

        rsi_val = rsi_arr[-1]
        rsi_prev = rsi_arr[-2]
        macd_h = macd_hist[-1]
        macd_prev = macd_hist[-2]
        k_val = k[-1]
        d_val = d_slow[-1]

        # Stoch triggers (histórico)
        lags = self.cfg.get("stoch_lags", 25)
        recent_k = k[-lags:]
        recent_d = d_slow[-lags:]
        trigger_long = np.any((recent_k < 20) & (recent_d < 20))
        trigger_short = np.any((recent_k > 80) & (recent_d > 80))

        # Condições osciladores actuais
        k_mid = 20 < k_val < 80 and 20 < d_val < 80
        rsi_bull = rsi_val > 50 and rsi_val > rsi_prev
        rsi_bear = rsi_val < 50 and rsi_val < rsi_prev
        macd_bull = macd_h > 0 and macd_prev <= 0
        macd_bear = macd_h < 0 and macd_prev >= 0

        long_cond = (
            ema_bull
            and above_vwap
            and trigger_long
            and k_mid
            and rsi_bull
            and macd_bull
        )

        short_cond = (
            ema_bear
            and below_vwap
            and trigger_short
            and k_mid
            and rsi_bear
            and macd_bear
        )

        if not (long_cond or short_cond):
            return {"signal": 0, "reason": "sem sinal confluente"}

        # SL/TP por multiplicador fixo
        sl_mult = self.cfg["sl_mult"]
        tp_mult = self.cfg["tp_mult"]

        if long_cond:
            sl = round(price * sl_mult, 2)
            tp = round(price * tp_mult, 2)
            reason = []
            if ema_bull: reason.append("EMA trend bullish")
            if above_vwap: reason.append("acima VWAP")
            if trigger_long: reason.append("Stoch oversold recente")
            if k_mid: reason.append("Stoch K/D 20-80")
            if rsi_bull: reason.append("RSI > 50 e a subir")
            if macd_bull: reason.append("MACD hist cruzou +")
            return {
                "signal": 1,
                "reason": " | ".join(reason),
                "sl": sl,
                "tp": tp,
            }

        if short_cond:
            sl = round(price * (2 - sl_mult), 2) if sl_mult < 1 else round(price * sl_mult, 2)
            # mais simples: usar tp_mult para baixo
            sl = round(price * (1 + (1 - sl_mult)), 2) if sl_mult < 1 else round(price * sl_mult, 2)
            tp = round(price * (2 - tp_mult), 2) if tp_mult > 1 else round(price * tp_mult, 2)
            reason = []
            if ema_bear: reason.append("EMA trend bearish")
            if below_vwap: reason.append("abaixo VWAP")
            if trigger_short: reason.append("Stoch overbought recente")
            if k_mid: reason.append("Stoch K/D 20-80")
            if rsi_bear: reason.append("RSI < 50 e a descer")
            if macd_bear: reason.append("MACD hist cruzou -")
            return {
                "signal": -1,
                "reason": " | ".join(reason),
                "sl": sl,
                "tp": tp,
            }

        return {"signal": 0, "reason": "sem sinal"}


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


class StochBot:
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
        log.info("  STOCH-RSI-MACD BOT INICIADO")
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

        data = self.hl.get_candles(CFG["timeframe"], limit=self.cfg.get("lookback", 200))
        if len(data["close"]) < 80:
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

        is_buy = sig["signal"] == 1
        self.hl.place_market(is_buy, size)
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
        period = tf_secs.get(CFG["timeframe"], 300)
        now = time.time()
        wait = period - (now % period) + 2
        log.info(f"Próxima vela em {wait:.0f}s ({CFG['timeframe']})")
        time.sleep(wait)


if __name__ == "__main__":
    bot = StochBot()
    bot.run()
