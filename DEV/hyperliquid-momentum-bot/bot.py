#!/usr/bin/env python3
"""
HL-MOMENTUM-BOT
Hyperliquid Futures Trend + Momentum Bot

Fibonacci EMA stack: 8 / 21 / 55 / 89 / 233
  EMA  8  -> short-term momentum (~2h)
  EMA 21  -> short/medium trend  (~5h)
  EMA 55  -> medium trend        (~14h)
  EMA 89  -> medium/long trend   (~22h)
  EMA 233 -> major trend         (~58h)

Edge:
  * Full Fibonacci cascade + VWAP
  * Pullback confirmation: RSI(14), MACD(12,26,9), StochRSI(14,3,3)
  * Risk engine: ATR(14) stop-loss, account-risk position sizing
"""

import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(divide="ignore", invalid="ignore")

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MONITOR_INTERVAL_SECONDS = 10

TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_FILE}")
    with open(CONFIG_FILE, encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calc_ema(series: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    out = np.full(len(series), np.nan, dtype=float)
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def calc_rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.empty(len(series), dtype=float)
    avg_loss = np.empty(len(series), dtype=float)
    avg_gain[:period] = avg_loss[:period] = 0
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(series)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out


def calc_macd(
    series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple:
    macd_line = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_stoch_rsi(
    series: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple:
    r = calc_rsi(series, rsi_period)
    k_line = np.full(len(series), np.nan, dtype=float)
    d_line = np.full(len(series), np.nan, dtype=float)
    for i in range(stoch_period, len(series)):
        window = r[i - stoch_period + 1:i + 1]
        low, high = np.nanmin(window), np.nanmax(window)
        k_line[i] = 0.5 if high == low else (r[i] - low) / (high - low)
        if i >= stoch_period + smooth_k - 1:
            d_line[i] = np.nanmean(k_line[i - smooth_k + 1:i + 1])
    return k_line, d_line


def calc_atr(highs, lows, closes, period: int = 14) -> np.ndarray:
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    true_range = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    out = np.full(len(true_range), np.nan, dtype=float)
    out[period - 1] = np.mean(true_range[:period])
    for i in range(period, len(true_range)):
        out[i] = (out[i - 1] * (period - 1) + true_range[i]) / period
    return out


def calc_vwap(highs, lows, closes, volumes) -> np.ndarray:
    typical_price = (highs + lows + closes) / 3
    cum_volume = np.cumsum(volumes)
    cum_tp_volume = np.cumsum(typical_price * volumes)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cum_volume == 0, np.nan, cum_tp_volume / cum_volume)


def compute_indicators(data: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, Any]:
    highs = data["high"]
    lows = data["low"]
    closes = data["close"]
    volumes = data["volume"]

    e8 = calc_ema(closes, cfg["ema_8"])
    e21 = calc_ema(closes, cfg["ema_21"])
    e55 = calc_ema(closes, cfg["ema_55"])
    e89 = calc_ema(closes, cfg["ema_89"])
    e233 = calc_ema(closes, cfg["ema_233"])
    rsi_values = calc_rsi(closes, cfg["rsi_length"])
    _, _, macd_hist = calc_macd(closes, 12, 26, 9)
    stoch_k, _ = calc_stoch_rsi(closes, 14, 14, 3, 3)
    atr_values = calc_atr(highs, lows, closes, 14)
    vwap_values = calc_vwap(highs, lows, closes, volumes)

    return {
        "price": float(closes[-1]),
        "ema_8": float(e8[-1]),
        "ema_21": float(e21[-1]),
        "ema_55": float(e55[-1]),
        "ema_89": float(e89[-1]),
        "ema_233": float(e233[-1]),
        "rsi": float(rsi_values[-1]),
        "rsi_prev": float(rsi_values[-2]),
        "macd_hist": float(macd_hist[-1]),
        "macd_hist_prev": float(macd_hist[-2]),
        "stoch_k": float(stoch_k[-1]),
        "stoch_k_prev": float(stoch_k[-2]),
        "atr": float(atr_values[-1]),
        "vwap": float(vwap_values[-1]),
    }


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

def evaluate_cascade(ind: Dict[str, Any]) -> tuple:
    """Evaluate each Fibonacci cascade link individually."""
    p = ind["price"]
    vw = ind["vwap"]
    e8, e21, e55, e89, e233 = ind["ema_8"], ind["ema_21"], ind["ema_55"], ind["ema_89"], ind["ema_233"]

    bull_links = [
        (p > vw,      f"{p:.2f} > VWAP {vw:.2f}"),
        (e8 > e21,    f"EMA8 {e8:.2f} > EMA21 {e21:.2f}"),
        (e21 > e55,   f"EMA21 {e21:.2f} > EMA55 {e55:.2f}"),
        (e55 > e89,   f"EMA55 {e55:.2f} > EMA89 {e89:.2f}"),
        (e89 > e233,  f"EMA89 {e89:.2f} > EMA233 {e233:.2f}"),
    ]
    bear_links = [
        (p < vw,      f"{p:.2f} < VWAP {vw:.2f}"),
        (e8 < e21,    f"EMA8 {e8:.2f} < EMA21 {e21:.2f}"),
        (e21 < e55,   f"EMA21 {e21:.2f} < EMA55 {e55:.2f}"),
        (e55 < e89,   f"EMA55 {e55:.2f} < EMA89 {e89:.2f}"),
        (e89 < e233,  f"EMA89 {e89:.2f} < EMA233 {e233:.2f}"),
    ]
    bull_ok = all(ok for ok, _ in bull_links)
    bear_ok = all(ok for ok, _ in bear_links)
    return bull_ok, bear_ok, bull_links


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

def print_monitor(
    ind: Dict[str, Any],
    position: Optional[Dict],
    cfg: Dict[str, Any],
    next_candle_in: float,
) -> None:
    p = ind["price"]
    vw = ind["vwap"]
    e8, e21, e55, e89, e233 = ind["ema_8"], ind["ema_21"], ind["ema_55"], ind["ema_89"], ind["ema_233"]
    r = ind["rsi"]
    mh = ind["macd_hist"]
    sk = ind["stoch_k"]
    a = ind["atr"]

    bull_ok, bear_ok, bull_links = evaluate_cascade(ind)
    links_ok = sum(1 for ok, _ in bull_links if ok)

    if bull_ok:
        trend_str = "\u2b06\ufe0f  BULLISH  \u2705 8>21>55>89>233"
    elif bear_ok:
        trend_str = "\u2b07\ufe0f  BEARISH  \u2705 8<21<55<89<233"
    else:
        trend_str = f"\u27a1\ufe0f  LATERAL  ({links_ok}/5 bullish links)"

    def tick(ok): return "\u2705" if ok else "\u274c"

    l1 = tick(bull_links[0][0])
    l2 = tick(bull_links[1][0])
    l3 = tick(bull_links[2][0])
    l4 = tick(bull_links[3][0])
    l5 = tick(bull_links[4][0])

    rsi_ok = tick(r > 45 and r > ind["rsi_prev"])
    macd_ok = tick(mh > 0 and mh > ind["macd_hist_prev"])
    stoch_ok = tick(ind["stoch_k_prev"] < 0.2 and sk > ind["stoch_k_prev"])

    all_long = bull_ok and rsi_ok == "\u2705" and macd_ok == "\u2705" and stoch_ok == "\u2705"
    signal_str = "\u26a1 LONG SIGNAL DETECTED!" if all_long else "   Waiting for signal..."

    position_str = "None"
    if position:
        pnl = p - position["entry"] if position["side"] == "long" else position["entry"] - p
        position_str = (
            f"{position['side'].upper()} | entry={position['entry']} "
            f"| size={position['size']} | PnL dist={pnl:+.2f}"
        )

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")

    print(f"""
\u250c{chr(9472)*61}\u2510
\u2502  HL-MOMENTUM-BOT  [{ts} UTC]  {cfg['symbol']}-PERP {cfg['timeframe']}          \u2502
\u251c{chr(9472)*61}\u2524
\u2502  \U0001f4b0 Price    : {p:<14.4f}   VWAP   : {vw:.4f}
\u2502  \U0001f4c8 Trend    : {trend_str}
\u251c{chr(9472)*61}\u2524
\u2502  FIBONACCI CASCADE  (bull: 8 > 21 > 55 > 89 > 233)
\u2502  {l1} Price  > VWAP   : {p:.2f} {'>' if p > vw else '<'} {vw:.2f}
\u2502  {l2} EMA8   > EMA21  : {e8:.2f} {'>' if e8 > e21 else '<'} {e21:.2f}
\u2502  {l3} EMA21  > EMA55  : {e21:.2f} {'>' if e21 > e55 else '<'} {e55:.2f}
\u2502  {l4} EMA55  > EMA89  : {e55:.2f} {'>' if e55 > e89 else '<'} {e89:.2f}
\u2502  {l5} EMA89  > EMA233 : {e89:.2f} {'>' if e89 > e233 else '<'} {e233:.2f}
\u251c{chr(9472)*61}\u2524
\u2502  MOMENTUM CONFIRMATION
\u2502  {rsi_ok} RSI(14)      : {r:.1f}  (prev: {ind['rsi_prev']:.1f})   [> 45 and rising]
\u2502  {macd_ok} MACD hist   : {mh:.5f}  (prev: {ind['macd_hist_prev']:.5f})  [> 0 and rising]
\u2502  {stoch_ok} StochRSI K : {sk:.3f}  (prev: {ind['stoch_k_prev']:.3f})  [left < 0.20]
\u2502  \U0001f4ca ATR(14)     : {a:.4f}
\u251c{chr(9472)*61}\u2524
\u2502  \U0001f4e6 Position  : {position_str}
\u2502  {signal_str}
\u2502  \u23f1\ufe0f  Next candle in : {next_candle_in:.0f}s
\u2514{chr(9472)*61}\u2518""", flush=True)


# ---------------------------------------------------------------------------
# Hyperliquid Client
# ---------------------------------------------------------------------------

class HLClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.symbol = cfg["symbol"]
        self.testnet = cfg.get("testnet", True)
        url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(url, skip_ws=True)
        wallet = eth_account.Account.from_key(cfg["secret_key"])
        self.exchange = Exchange(wallet, url, account_address=cfg["account_address"])
        mode = "TESTNET" if self.testnet else "MAINNET"
        log.info(f"HLClient initialized | {mode} | {self.symbol}")

    def get_candles(self, interval: str, limit: int = 600) -> Dict[str, np.ndarray]:
        now_ms = int(time.time() * 1000)
        since = now_ms - limit * TIMEFRAME_SECONDS.get(interval, 900) * 1000
        request = {
            "type": "candleSnapshot",
            "req": {
                "coin": self.symbol,
                "interval": interval,
                "startTime": since,
                "endTime": now_ms,
            },
        }
        raw = self.info.post("/info", request)
        opens, highs, lows, closes, volumes, times = [], [], [], [], [], []
        for candle in raw:
            opens.append(float(candle["o"]))
            highs.append(float(candle["h"]))
            lows.append(float(candle["l"]))
            closes.append(float(candle["c"]))
            volumes.append(float(candle["v"]))
            times.append(int(candle["t"]))
        return {
            "open": np.array(opens, dtype=float),
            "high": np.array(highs, dtype=float),
            "low": np.array(lows, dtype=float),
            "close": np.array(closes, dtype=float),
            "volume": np.array(volumes, dtype=float),
            "time": np.array(times),
        }

    def get_balance(self) -> float:
        state = self.info.user_state(CFG["account_address"])
        return float(state["crossMarginSummary"]["accountValue"])

    def get_open_position(self) -> Optional[Dict[str, Any]]:
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

    def set_leverage(self, leverage: int) -> None:
        try:
            self.exchange.update_leverage(leverage, self.symbol, is_cross=True)
            log.info(f"Leverage set: {leverage}x")
        except Exception as exc:
            log.warning(f"Failed to set leverage: {exc}")

    def _get_mid_price(self) -> float:
        return float(self.info.all_mids()[self.symbol])

    def place_market_order(self, is_buy: bool, size: float):
        price = round(self._get_mid_price() * (1.0015 if is_buy else 0.9985), 2)
        result = self.exchange.order(
            self.symbol, is_buy, size, price,
            {"limit": {"tif": "Ioc"}}, reduce_only=False,
        )
        action = "BUY" if is_buy else "SELL"
        log.info(f"{action} MARKET | size={size} | ~price={price} | {result}")
        return result

    def close_position(self, is_long: bool, size: float):
        price = round(self._get_mid_price() * (0.998 if is_long else 1.002), 2)
        result = self.exchange.order(
            self.symbol, not is_long, size, price,
            {"limit": {"tif": "Ioc"}}, reduce_only=True,
        )
        side = "LONG" if is_long else "SHORT"
        log.info(f"CLOSE {side} | size={size} | ~price={price} | {result}")
        return result

    def get_coin_decimals(self) -> int:
        for asset in self.info.meta()["universe"]:
            if asset["name"] == self.symbol:
                return asset.get("szDecimals", 3)
        return 3


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.risk_fraction = cfg["risk_pct"] / 100

    def calculate_position_size(self, balance: float, entry: float, stop_loss: float, decimals: int = 3) -> float:
        """Risk-based position sizing: risk_pct% of account balance per trade."""
        sl_distance = abs(entry - stop_loss) / entry
        if sl_distance == 0:
            return 0.0
        factor = 10 ** decimals
        raw_size = (balance * self.risk_fraction) / sl_distance / entry
        return max(math.floor(raw_size * factor) / factor, 10 ** (-decimals))


# ---------------------------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def analyse(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        closes = data["close"]
        min_candles = self.cfg["ema_233"] + 50
        if len(closes) < min_candles:
            return {"signal": 0, "reason": f"insufficient data ({len(closes)}/{min_candles} candles)"}

        ind = compute_indicators(data, self.cfg)

        required_keys = ["rsi", "macd_hist", "stoch_k", "atr", "vwap"]
        if any(math.isnan(ind[k]) for k in required_keys):
            return {"signal": 0, "reason": "incomplete indicators", "ind": ind}

        bull_ok, bear_ok, _ = evaluate_cascade(ind)
        atr_mult = self.cfg["atr_multiplier"]
        rr = self.cfg["reward_risk"]

        # --- Long ---
        long_conditions = (
            bull_ok
            and ind["rsi"] > 45
            and ind["rsi"] > ind["rsi_prev"]
            and ind["macd_hist"] > 0
            and ind["macd_hist"] > ind["macd_hist_prev"]
            and ind["stoch_k_prev"] < 0.2
            and ind["stoch_k"] > ind["stoch_k_prev"]
        )
        if long_conditions:
            risk = atr_mult * ind["atr"]
            return {
                "signal": 1,
                "side": "long",
                "sl": round(ind["price"] - risk, 2),
                "tp": round(ind["price"] + risk * rr, 2),
                "reason": "EMA8>21>55>89>233 | VWAP | RSI rising | MACD rising | StochRSI oversold->up",
                "ind": ind,
            }

        # --- Short (optional) ---
        if self.cfg.get("enable_shorts", False):
            short_conditions = (
                bear_ok
                and ind["rsi"] < 55
                and ind["rsi"] < ind["rsi_prev"]
                and ind["macd_hist"] < 0
                and ind["macd_hist"] < ind["macd_hist_prev"]
                and ind["stoch_k_prev"] > 0.8
                and ind["stoch_k"] < ind["stoch_k_prev"]
            )
            if short_conditions:
                risk = atr_mult * ind["atr"]
                return {
                    "signal": 1,
                    "side": "short",
                    "sl": round(ind["price"] + risk, 2),
                    "tp": round(ind["price"] - risk * rr, 2),
                    "reason": "EMA8<21<55<89<233 | VWAP | RSI falling | MACD falling | StochRSI overbought->down",
                    "ind": ind,
                }

        return {"signal": 0, "reason": "no confluence", "ind": ind}


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class MomentumBot:
    def __init__(self):
        self.cfg = CFG
        self.client = HLClient(self.cfg)
        self.signal_engine = SignalEngine(self.cfg)
        self.risk_manager = RiskManager(self.cfg)
        self.client.set_leverage(self.cfg["leverage"])
        self.coin_decimals = self.client.get_coin_decimals()
        self.active_sl: Optional[float] = None
        self.active_tp: Optional[float] = None

        log.info("=" * 70)
        log.info("  HL-MOMENTUM-BOT STARTED")
        log.info(f"  Symbol     : {self.cfg['symbol']}-PERP")
        log.info(f"  Timeframe  : {self.cfg['timeframe']}")
        log.info(f"  Leverage   : {self.cfg['leverage']}x")
        log.info(f"  Risk/trade : {self.cfg['risk_pct']}%")
        log.info("  EMA Stack  : 8 / 21 / 55 / 89 / 233")
        log.info("=" * 70)

    def run(self) -> None:
        log.info("Bot running... Press Ctrl+C to stop.")
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("Bot stopped by user.")
                break
            except Exception as exc:
                log.error(f"Main loop error: {exc}", exc_info=True)
                time.sleep(30)

    def _tick(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"{'─' * 50} {ts}")

        data = self.client.get_candles(self.cfg["timeframe"], self.cfg["lookback"])
        price = float(data["close"][-1])
        log.info(f"Price : {price}")

        position = self.client.get_open_position()
        if position:
            log.info(f"Open position: {position['side'].upper()} | size={position['size']} | entry={position['entry']}")
            self._manage_position(position, price)
            self._monitor_loop(data)
            return

        signal = self.signal_engine.analyse(data)
        log.info(f"Signal : {signal['signal']} | {signal.get('side', '-')} | {signal.get('reason', '')}")

        if signal["signal"] == 1:
            balance = self.client.get_balance()
            size = self.risk_manager.calculate_position_size(
                balance, price, signal["sl"], self.coin_decimals
            )
            log.info(f"Balance: {balance:.2f} | Size: {size} | SL: {signal['sl']} | TP: {signal['tp']}")
            if size > 0:
                self.client.place_market_order(signal["side"] == "long", size)
                self.active_sl = signal["sl"]
                self.active_tp = signal["tp"]
            else:
                log.warning("Calculated size is zero — order skipped.")

        self._monitor_loop(data)

    def _monitor_loop(self, data_snapshot: Dict[str, np.ndarray]) -> None:
        period = TIMEFRAME_SECONDS.get(self.cfg["timeframe"], 900)
        now = time.time()
        candle_end = now + (period - (now % period)) + 2
        min_candles = self.cfg["ema_233"] + 50
        has_indicators = len(data_snapshot["close"]) >= min_candles

        while True:
            remaining = candle_end - time.time()
            if remaining <= 0:
                break
            try:
                live_price = float(self.client.info.all_mids()[self.cfg["symbol"]])
                data_snapshot["close"][-1] = live_price
            except Exception:
                pass

            if has_indicators:
                try:
                    indicators = compute_indicators(data_snapshot, self.cfg)
                    position = self.client.get_open_position()
                    print_monitor(indicators, position, self.cfg, remaining)
                except Exception as exc:
                    log.debug(f"Monitor render error: {exc}")
            else:
                price = float(data_snapshot["close"][-1])
                print(
                    f"\r  \U0001f4b0 {self.cfg['symbol']} = {price}  "
                    f"| Loading indicators... next candle in {remaining:.0f}s",
                    end="", flush=True,
                )

            time.sleep(MONITOR_INTERVAL_SECONDS)

    def _manage_position(self, position: Dict[str, Any], price: float) -> None:
        if self.active_sl is None or self.active_tp is None:
            log.warning("SL/TP not set — cannot manage position.")
            return
        is_long = position["side"] == "long"
        hit_tp = (is_long and price >= self.active_tp) or (not is_long and price <= self.active_tp)
        hit_sl = (is_long and price <= self.active_sl) or (not is_long and price >= self.active_sl)
        if hit_tp:
            log.info(f"TAKE PROFIT hit @ {price} (TP={self.active_tp})")
            self.client.close_position(is_long, position["size"])
            self._reset_trade()
        elif hit_sl:
            log.info(f"STOP LOSS hit @ {price} (SL={self.active_sl})")
            self.client.close_position(is_long, position["size"])
            self._reset_trade()
        else:
            dist_tp = abs(price - self.active_tp)
            dist_sl = abs(price - self.active_sl)
            log.info(f"Position OK | dist_TP={dist_tp:.2f} | dist_SL={dist_sl:.2f}")

    def _reset_trade(self) -> None:
        self.active_sl = None
        self.active_tp = None


if __name__ == "__main__":
    bot = MomentumBot()
    bot.run()
