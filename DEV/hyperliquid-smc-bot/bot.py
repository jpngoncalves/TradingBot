#!/usr/bin/env python3
"""
=============================================================
  HL-SMC-BOT  |  Hyperliquid Futures Trading Bot
  Estratégia  :  Smart Money Concepts (SMC) + EMA Trend + RSI/MACD/StochRSI
  Autor       :  DEV / jpngoncalves
  Versão      :  1.1.0
  Timeframe   :  15m (curto/médio prazo)
=============================================================

ESTRATÉGIA RESUMIDA
-------------------
LONG quando:
  1. EMA8 > EMA21 > EMA55  (tendência bullish confirmada)
  2. PSAR abaixo do preço  (momentum bullish)
  3. CHoCH/BOS bullish detetado (estrutura quebrada para cima)
  4. Preço entra em Order Block bullish ou FVG bullish
  5. Preço acima do VWAP   (filtro institucional)
  6. BB: preço tocou banda inferior (oversold local)
  7. Pelo menos 2 destes 3 a favor do bull:
       - RSI(14) a sair de zona baixa (30–60, inclinado para cima)
       - MACD(12,26,9) histograma a cruzar para positivo
       - StochRSI(14,3,3) a sair de <0.2 para cima

SHORT quando:
  1. EMA8 < EMA21 < EMA55  (tendência bearish confirmada)
  2. PSAR acima do preço   (momentum bearish)
  3. CHoCH/BOS bearish detetado (estrutura quebrada para baixo)
  4. Preço entra em Order Block bearish ou FVG bearish
  5. Preço abaixo do VWAP  (filtro institucional)
  6. BB: preço tocou banda superior (overbought local)
  7. Pelo menos 2 destes 3 a favor do bear:
       - RSI(14) a sair de zona alta (70–40, inclinado para baixo)
       - MACD(12,26,9) histograma a cruzar para negativo
       - StochRSI(14,3,3) a sair de >0.8 para baixo

GESTÃO DE RISCO
---------------
  - Stop Loss  : ATR x 1.5 abaixo/acima do entry
  - Take Profit: Risk x 2.0 (R:R mínimo de 1:2)
  - Tamanho    : % do account em risco por trade (config)
  - Máx. posições simultâneas: 1
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

# ── Hyperliquid SDK ──
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# ─────────────────────────────────────────
#  CONFIGURAÇÃO & LOGGING
# ─────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
CFG_FILE  = BASE_DIR / "config.json"
LOG_FILE  = BASE_DIR / "logs" / "bot.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def load_config() -> dict:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"config.json não encontrado em {CFG_FILE}")
    with open(CFG_FILE) as f:
        return json.load(f)

CFG = load_config()

logging.basicConfig(
    level   = getattr(logging, CFG.get("log_level", "INFO")),
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("HL-SMC-BOT")

# ─────────────────────────────────────────
#  INDICADORES (Python-native, sem talib)
# ─────────────────────────────────────────
def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    k   = 2 / (period + 1)
    out = np.full_like(series, np.nan)
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def bollinger_bands(series: np.ndarray, length: int = 21, std_mult: float = 2.0):
    """Returns (basis, upper, lower)."""
    basis = np.array([np.mean(series[max(0, i - length + 1):i + 1]) for i in range(len(series))])
    std   = np.array([np.std(series[max(0, i - length + 1):i + 1])  for i in range(len(series))])
    return basis, basis + std_mult * std, basis - std_mult * std


def psar(highs, lows, start=0.02, increment=0.02, maximum=0.2):
    """Parabolic SAR — returns array."""
    n       = len(highs)
    psar_v  = np.full(n, np.nan)
    bull    = True
    af      = start
    ep      = lows[0]
    hp      = highs[0]
    lp      = lows[0]

    for i in range(2, n):
        if bull:
            psar_v[i] = psar_v[i - 1] + af * (hp - psar_v[i - 1]) if not np.isnan(psar_v[i-1]) else lp
            if lows[i] < psar_v[i]:
                bull        = False
                psar_v[i]   = hp
                lp          = lows[i]
                af          = start
                ep          = lows[i]
            else:
                if highs[i] > hp:
                    hp  = highs[i]
                    af  = min(af + increment, maximum)
                    ep  = highs[i]
        else:
            psar_v[i] = psar_v[i - 1] + af * (lp - psar_v[i - 1]) if not np.isnan(psar_v[i-1]) else hp
            if highs[i] > psar_v[i]:
                bull        = True
                psar_v[i]   = lp
                hp          = highs[i]
                af          = start
                ep          = highs[i]
            else:
                if lows[i] < lp:
                    lp  = lows[i]
                    af  = min(af + increment, maximum)
                    ep  = lows[i]
    return psar_v, bull


def vwap(closes, highs, lows, volumes):
    """Session VWAP."""
    hlc3    = (highs + lows + closes) / 3
    cum_vol = np.cumsum(volumes)
    cum_pv  = np.cumsum(hlc3 * volumes)
    return cum_pv / np.where(cum_vol == 0, 1, cum_vol)


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    """Classic RSI."""
    delta = np.diff(series)
    up    = np.where(delta > 0, delta, 0.0)
    down  = np.where(delta < 0, -delta, 0.0)
    roll_up  = np.empty_like(series)
    roll_down= np.empty_like(series)
    roll_up[:period]   = 0
    roll_down[:period] = 0
    roll_up[period]    = np.mean(up[:period])
    roll_down[period]  = np.mean(down[:period])
    for i in range(period + 1, len(series)):
        roll_up[i]   = (roll_up[i-1] * (period - 1) + up[i-1]) / period
        roll_down[i] = (roll_down[i-1] * (period - 1) + down[i-1]) / period
    rs  = np.where(roll_down == 0, np.inf, roll_up / roll_down)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def macd(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stoch_rsi(series: np.ndarray, rsi_length: int = 14, stoch_length: int = 14,
              k: int = 3, d: int = 3):
    """StochRSI (returns %K, %D in [0,1])."""
    r = rsi(series, rsi_length)
    out_k = np.full_like(series, np.nan, dtype=float)
    out_d = np.full_like(series, np.nan, dtype=float)
    for i in range(stoch_length, len(series)):
        window = r[i - stoch_length + 1:i + 1]
        r_min  = np.nanmin(window)
        r_max  = np.nanmax(window)
        if r_max - r_min == 0:
            k_val = 0.5
        else:
            k_val = (r[i] - r_min) / (r_max - r_min)
        out_k[i] = k_val
        if i >= stoch_length + k - 1:
            out_d[i] = np.nanmean(out_k[i - k + 1:i + 1])
    return out_k, out_d


def detect_swing_points(highs, lows, lookback: int = 50):
    """Detect swing highs and lows (BOS/CHoCH proxy)."""
    n           = len(highs)
    swing_highs = []
    swing_lows  = []
    for i in range(lookback, n - 1):
        window_h = highs[max(0, i - lookback):i]
        window_l = lows [max(0, i - lookback):i]
        if highs[i] == max(window_h):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(window_l):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def detect_structure_break(closes, swing_highs, swing_lows):
    """
    CHoCH/BOS detection:
      BULLISH  = price crosses above last swing high
      BEARISH  = price crosses below last swing low
    Returns: 1 (bullish break), -1 (bearish break), 0 (none)
    """
    if not swing_highs or not swing_lows:
        return 0
    last_sh = swing_highs[-1][1]
    last_sl = swing_lows[-1][1]
    c       = closes[-1]
    c1      = closes[-2]
    if c1 <= last_sh and c > last_sh:
        return 1
    if c1 >= last_sl and c < last_sl:
        return -1
    return 0


def detect_fvg(opens, highs, lows, closes):
    """
    Fair Value Gap:
      Bullish FVG: candle[i-2].high < candle[i].low
      Bearish FVG: candle[i-2].low  > candle[i].high
    Returns: (bias, top, bottom) or (0, None, None)
    """
    if len(closes) < 3:
        return 0, None, None
    if lows[-1] > highs[-3]:   # bullish gap
        return 1, lows[-1], highs[-3]
    if highs[-1] < lows[-3]:   # bearish gap
        return -1, lows[-3], highs[-1]
    return 0, None, None


def detect_order_block(opens, highs, lows, closes, lookback: int = 10):
    """
    Order Block (simplified):
      Bullish OB: last bearish candle before a strong bullish move
      Bearish OB: last bullish candle before a strong bearish move
    Returns: (bias, ob_high, ob_low) or (0, None, None)
    """
    n = len(closes)
    if n < lookback + 2:
        return 0, None, None
    body    = abs(closes[-1] - opens[-1])
    atr_est = np.mean(highs[-lookback:] - lows[-lookback:])
    if body < atr_est * 0.8:
        return 0, None, None
    if closes[-1] > opens[-1]:
        for i in range(2, lookback):
            if closes[-i] < opens[-i]:
                return 1, highs[-i], lows[-i]
    else:
        for i in range(2, lookback):
            if closes[-i] > opens[-i]:
                return -1, highs[-i], lows[-i]
    return 0, None, None


def atr(highs, lows, closes, period: int = 14) -> float:
    """Average True Range — last value."""
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(abs(highs[1:] - closes[:-1]),
                    abs(lows[1:] - closes[:-1])))
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


# ─────────────────────────────────────────
#  HYPERLIQUID CLIENT
# ─────────────────────────────────────────
class HLClient:
    """Thin wrapper around the Hyperliquid SDK."""

    def __init__(self, cfg: dict):
        self.symbol  = cfg["symbol"]
        self.testnet = cfg.get("testnet", True)
        url          = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info    = Info(url, skip_ws=True)
        wallet       = eth_account.Account.from_key(cfg["secret_key"])
        self.exchange= Exchange(wallet, url, account_address=cfg["account_address"])
        log.info(f"Cliente HL iniciado | {'TESTNET' if self.testnet else 'MAINNET'} | {self.symbol}")

    def get_candles(self, interval: str = "15m", limit: int = 300) -> dict:
        """Fetch OHLCV data. Returns dict of np.arrays."""
        now_ms  = int(time.time() * 1000)
        tf_secs = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
                   "1h": 3600, "4h": 14400, "1d": 86400}
        tf_ms   = tf_secs.get(interval, 900) * 1000
        since   = now_ms - limit * tf_ms
        req     = {"type": "candleSnapshot", "req": {
                   "coin": self.symbol, "interval": interval,
                   "startTime": since, "endTime": now_ms}}
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
            "open": np.array(o), "high": np.array(h),
            "low":  np.array(l), "close": np.array(c),
            "volume": np.array(v), "time": np.array(t),
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
                        "side":  "long" if size > 0 else "short",
                        "size":  abs(size),
                        "entry": float(pos["position"]["entryPx"]),
                    }
        return None

    def set_leverage(self, lev: int):
        try:
            self.exchange.update_leverage(lev, self.symbol, is_cross=True)
            log.info(f"Leverage definida: {lev}x")
        except Exception as e:
            log.warning(f"Erro ao definir leverage: {e}")

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

    def place_limit(self, is_buy: bool, size: float, price: float) -> dict:
        result = self.exchange.order(
            self.symbol, is_buy, size, round(price, 2),
            {"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )
        log.info(f"{'BUY' if is_buy else 'SELL'} LIMIT | size={size} | px={price} | {result}")
        return result

    def close_position(self, is_buy: bool, size: float) -> dict:
        price = self._mid_price()
        slippage = 0.002
        px = price * (1 - slippage) if is_buy else price * (1 + slippage)
        px = round(px, 2)
        result = self.exchange.order(
            self.symbol, not is_buy, size, px,
            {"limit": {"tif": "Ioc"}},
            reduce_only=True,
        )
        log.info(f"CLOSE {'LONG' if is_buy else 'SHORT'} | size={size} | ~px={px} | {result}")
        return result

    def _mid_price(self) -> float:
        mids = self.info.all_mids()
        return float(mids[self.symbol])

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
    """Gera sinais LONG / SHORT / HOLD com base na estratégia SMC+EMA+Osciladores."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def analyse(self, d: dict) -> dict:
        o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
        n = len(c)
        if n < 260:
            return {"signal": 0, "reason": "dados insuficientes"}

        e8   = ema(c, self.cfg["ema_fast"])
        e21  = ema(c, self.cfg["ema_slow"])
        e55  = ema(c, self.cfg["ema_trend"])
        e233 = ema(c, self.cfg["ema_macro"])
        bb_b, bb_u, bb_l = bollinger_bands(c, self.cfg["bb_length"], self.cfg["bb_std"])
        psar_arr, psar_bull = psar(h, l)
        vwap_arr   = vwap(c, h, l, v)
        rsi_arr    = rsi(c, 14)
        macd_line, macd_signal, macd_hist = macd(c, 12, 26, 9)
        stoch_k, stoch_d = stoch_rsi(c, 14, 14, 3, 3)
        atr_val    = atr(h, l, c, 14)
        swing_h, swing_l = detect_swing_points(h, l, self.cfg["swing_lookback"])
        struct_brk = detect_structure_break(c, swing_h, swing_l)
        fvg_bias, fvg_top, fvg_bot = detect_fvg(o, h, l, c)
        ob_bias, ob_h, ob_l        = detect_order_block(o, h, l, c)

        price = float(c[-1])

        trend_bull = e8[-1] > e21[-1] > e55[-1]
        trend_bear = e8[-1] < e21[-1] < e55[-1]

        psar_bullish = price > psar_arr[-1]
        psar_bearish = price < psar_arr[-1]

        above_vwap = price > vwap_arr[-1]
        below_vwap = price < vwap_arr[-1]

        bb_oversold   = price <= bb_l[-1]
        bb_overbought = price >= bb_u[-1]

        in_bull_ob  = ob_bias == 1  and ob_l is not None and ob_l <= price <= ob_h
        in_bear_ob  = ob_bias == -1 and ob_l is not None and ob_l <= price <= ob_h
        in_bull_fvg = fvg_bias == 1  and fvg_bot is not None and fvg_bot <= price <= fvg_top
        in_bear_fvg = fvg_bias == -1 and fvg_bot is not None and fvg_bot <= price <= fvg_top
        in_bull_zone = in_bull_ob or in_bull_fvg or bb_oversold
        in_bear_zone = in_bear_ob or in_bear_fvg or bb_overbought

        # Osciladores
        rsi_val  = rsi_arr[-1]
        rsi_prev = rsi_arr[-2]
        macd_h   = macd_hist[-1]
        macd_prev= macd_hist[-2]
        k_val    = stoch_k[-1]
        k_prev   = stoch_k[-2]
        d_val    = stoch_d[-1]

        rsi_bull  = rsi_prev < 40 and 30 <= rsi_val <= 60 and rsi_val > rsi_prev
        rsi_bear  = rsi_prev > 60 and 40 <= rsi_val <= 70 and rsi_val < rsi_prev

        macd_bull = macd_prev <= 0 and macd_h > 0
        macd_bear = macd_prev >= 0 and macd_h < 0

        stoch_bull = k_prev < 0.2 and k_val > 0.2 and k_val > d_val
        stoch_bear = k_prev > 0.8 and k_val < 0.8 and k_val < d_val

        bull_count = sum([rsi_bull, macd_bull, stoch_bull])
        bear_count = sum([rsi_bear, macd_bear, stoch_bear])

        long_cond = (
            trend_bull
            and psar_bullish
            and above_vwap
            and (struct_brk == 1 or in_bull_zone)
            and bull_count >= 2
        )

        short_cond = (
            trend_bear
            and psar_bearish
            and below_vwap
            and (struct_brk == -1 or in_bear_zone)
            and bear_count >= 2
        )

        sl_dist = atr_val * 1.5
        tp_dist = sl_dist * 2.0

        if long_cond:
            reasons = []
            if trend_bull:    reasons.append("EMA trend bullish")
            if psar_bullish:  reasons.append("PSAR abaixo")
            if above_vwap:    reasons.append("acima VWAP")
            if struct_brk==1: reasons.append("BOS/CHoCH bullish")
            if in_bull_ob:    reasons.append("Order Block bullish")
            if in_bull_fvg:   reasons.append("FVG bullish")
            if bb_oversold:   reasons.append("BB oversold")
            if rsi_bull:      reasons.append("RSI bullish")
            if macd_bull:     reasons.append("MACD bullish")
            if stoch_bull:    reasons.append("StochRSI bullish")
            return {
                "signal":  1,
                "reason":  " | ".join(reasons),
                "sl":      round(price - sl_dist, 2),
                "tp":      round(price + tp_dist, 2),
                "atr_val": round(atr_val, 4),
            }

        if short_cond:
            reasons = []
            if trend_bear:     reasons.append("EMA trend bearish")
            if psar_bearish:   reasons.append("PSAR acima")
            if below_vwap:     reasons.append("abaixo VWAP")
            if struct_brk==-1: reasons.append("BOS/CHoCH bearish")
            if in_bear_ob:     reasons.append("Order Block bearish")
            if in_bear_fvg:    reasons.append("FVG bearish")
            if bb_overbought:  reasons.append("BB overbought")
            if rsi_bear:       reasons.append("RSI bearish")
            if macd_bear:      reasons.append("MACD bearish")
            if stoch_bear:     reasons.append("StochRSI bearish")
            return {
                "signal": -1,
                "reason": " | ".join(reasons),
                "sl":     round(price + sl_dist, 2),
                "tp":     round(price - tp_dist, 2),
                "atr_val":round(atr_val, 4),
            }

        return {"signal": 0, "reason": "sem sinal confluente", "atr_val": round(atr_val, 4)}


# ─────────────────────────────────────────
#  RISK MANAGER
# ─────────────────────────────────────────
class RiskManager:
    def __init__(self, cfg: dict):
        self.risk_pct = cfg["risk_pct"] / 100
        self.leverage = cfg["leverage"]

    def position_size(self, balance: float, entry: float, sl: float, decimals: int = 3) -> float:
        """Calcula tamanho da posição baseado no risco (% do account)."""
        risk_usd = balance * self.risk_pct
        dist_pct  = abs(entry - sl) / entry
        if dist_pct == 0:
            return 0.0
        size = (risk_usd / dist_pct) / entry
        factor = 10 ** decimals
        size = math.floor(size * factor) / factor
        return max(size, 10 ** (-decimals))


# ─────────────────────────────────────────
#  BOT PRINCIPAL
# ─────────────────────────────────────────
class SMCBot:
    def __init__(self):
        self.cfg    = CFG
        self.hl     = HLClient(CFG)
        self.engine = SignalEngine(CFG)
        self.risk   = RiskManager(CFG)
        self.hl.set_leverage(CFG["leverage"])
        self.decimals = self.hl.coin_decimals()
        self.active_sl: float | None = None
        self.active_tp: float | None = None
        log.info("=" * 55)
        log.info("  SMC BOT INICIADO")
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

        data = self.hl.get_candles(CFG["timeframe"], limit=300)
        if len(data["close"]) < 260:
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
        hit_sl  = price <= self.active_sl if is_long else price >= self.active_sl
        hit_tp  = price >= self.active_tp if is_long else price <= self.active_tp

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
        now    = time.time()
        wait   = period - (now % period) + 2
        log.info(f"Próxima vela em {wait:.0f}s ({CFG['timeframe']})")
        time.sleep(wait)


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    bot = SMCBot()
    bot.run()
