#!/usr/bin/env python3
"""
HL-MOMENTUM-BOT — Hyperliquid Futures Trend + Momentum Bot

Full Fibonacci EMA stack: 8 / 21 / 55 / 89 / 233
  EMA 8   → short-term momentum (~2h)
  EMA 21  → short/medium trend (~5h)
  EMA 55  → medium trend (~14h)
  EMA 89  → medium/long trend (~22h)
  EMA 233 → major trend (~58h)

Edge:
  * Full Fibonacci cascade + VWAP
  * Pullback confirmation: RSI(14), MACD(12,26,9), StochRSI(14,3,3)
  * Risk engine: ATR(14) for SL/TP and account-risk position sizing
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
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MONITOR_INTERVAL = 10


def load_config() -> Dict[str, Any]:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"config.json not found at {CFG_FILE}")
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


def ema(series: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    out = np.full(len(series), np.nan, dtype=float)
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = np.empty(len(series), dtype=float)
    roll_down = np.empty(len(series), dtype=float)
    roll_up[:period] = roll_down[:period] = 0
    roll_up[period] = np.mean(up[:period])
    roll_down[period] = np.mean(down[:period])
    for i in range(period + 1, len(series)):
        roll_up[i] = (roll_up[i - 1] * (period - 1) + up[i - 1]) / period
        roll_down[i] = (roll_down[i - 1] * (period - 1) + down[i - 1]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(roll_down == 0, np.inf, roll_up / roll_down)
        out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out


def macd(series: np.ndarray, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def stoch_rsi(series: np.ndarray, rsi_len=14, stoch_len=14, k=3, d=3):
    r = rsi(series, rsi_len)
    out_k = np.full(len(series), np.nan, dtype=float)
    out_d = np.full(len(series), np.nan, dtype=float)
    for i in range(stoch_len, len(series)):
        window = r[i - stoch_len + 1:i + 1]
        low, high = np.nanmin(window), np.nanmax(window)
        out_k[i] = 0.5 if high == low else (r[i] - low) / (high - low)
        if i >= stoch_len + k - 1:
            out_d[i] = np.nanmean(out_k[i - k + 1:i + 1])
    return out_k, out_d


def atr(highs, lows, closes, period=14) -> np.ndarray:
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    out = np.full(len(tr), np.nan, dtype=float)
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def vwap(highs, lows, closes, volumes) -> np.ndarray:
    typical_price = (highs + lows + closes) / 3
    cumulative_volume = np.cumsum(volumes)
    cumulative_tp_volume = np.cumsum(typical_price * volumes)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cumulative_volume == 0, np.nan, cumulative_tp_volume / cumulative_volume)


def compute_indicators(data: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, Any]:
    highs, lows, closes, volumes = data["high"], data["low"], data["close"], data["volume"]

    ema8 = ema(closes, cfg.get("ema_8", 8))
    ema21 = ema(closes, cfg.get("ema_fast", 21))
    ema55 = ema(closes, cfg.get("ema_slow", 55))
    ema89 = ema(closes, cfg.get("ema_89", 89))
    ema233 = ema(closes, cfg.get("ema_trend", 233))
    r = rsi(closes, cfg.get("rsi_length", 14))
    _, _, macd_hist = macd(closes, 12, 26, 9)
    stoch_k, _ = stoch_rsi(closes, 14, 14, 3, 3)
    a = atr(highs, lows, closes, 14)
    vw = vwap(highs, lows, closes, volumes)

    return {
        "price": float(closes[-1]),
        "ema8": float(ema8[-1]),
        "ema_fast": float(ema21[-1]),
        "ema_slow": float(ema55[-1]),
        "ema89": float(ema89[-1]),
        "ema_trend": float(ema233[-1]),
        "rsi": float(r[-1]),
        "rsi_prev": float(r[-2]),
        "macd_hist": float(macd_hist[-1]),
        "macd_prev": float(macd_hist[-2]),
        "stoch_k": float(stoch_k[-1]),
        "stoch_prev": float(stoch_k[-2]),
        "atr": float(a[-1]),
        "vwap": float(vw[-1]),
    }


def cascade_status(ind: Dict[str, Any]) -> tuple:
    price = ind["price"]
    vwap_value = ind["vwap"]
    ema8 = ind["ema8"]
    ema21 = ind["ema_fast"]
    ema55 = ind["ema_slow"]
    ema89 = ind["ema89"]
    ema233 = ind["ema_trend"]

    bull_links = [
        (price > vwap_value, f"Price {price:.1f} > VWAP {vwap_value:.1f}"),
        (ema8 > ema21, f"EMA8 {ema8:.1f} > EMA21 {ema21:.1f}"),
        (ema21 > ema55, f"EMA21 {ema21:.1f} > EMA55 {ema55:.1f}"),
        (ema55 > ema89, f"EMA55 {ema55:.1f} > EMA89 {ema89:.1f}"),
        (ema89 > ema233, f"EMA89 {ema89:.1f} > EMA233 {ema233:.1f}"),
    ]
    bear_links = [
        (price < vwap_value, f"Price {price:.1f} < VWAP {vwap_value:.1f}"),
        (ema8 < ema21, f"EMA8 {ema8:.1f} < EMA21 {ema21:.1f}"),
        (ema21 < ema55, f"EMA21 {ema21:.1f} < EMA55 {ema55:.1f}"),
        (ema55 < ema89, f"EMA55 {ema55:.1f} < EMA89 {ema89:.1f}"),
        (ema89 < ema233, f"EMA89 {ema89:.1f} < EMA233 {ema233:.1f}"),
    ]
    bull_ok = all(ok for ok, _ in bull_links)
    bear_ok = all(ok for ok, _ in bear_links)
    return bull_ok, bear_ok, bull_links


def print_monitor(ind: Dict[str, Any], position: Optional[Dict], cfg: Dict[str, Any], next_candle_in: float):
    price = ind["price"]
    vwap_value = ind["vwap"]
    ema8 = ind["ema8"]
    ema21 = ind["ema_fast"]
    ema55 = ind["ema_slow"]
    ema89 = ind["ema89"]
    ema233 = ind["ema_trend"]
    r = ind["rsi"]
    macd_hist = ind["macd_hist"]
    stoch_k = ind["stoch_k"]
    a = ind["atr"]

    bull_ok, bear_ok, bull_links = cascade_status(ind)

    if bull_ok:
        trend_str = "⬆️  BULLISH  ✅ 8>21>55>89>233 cascade"
    elif bear_ok:
        trend_str = "⬇️  BEARISH  ✅ 8<21<55<89<233 cascade"
    else:
        links_ok = sum(1 for ok, _ in bull_links if ok)
        trend_str = f"➡️  LATERAL  ({links_ok}/5 bullish links)"

    link1 = "✅" if bull_links[0][0] else "❌"
    link2 = "✅" if bull_links[1][0] else "❌"
    link3 = "✅" if bull_links[2][0] else "❌"
    link4 = "✅" if bull_links[3][0] else "❌"
    link5 = "✅" if bull_links[4][0] else "❌"

    rsi_ok = "✅" if r > 45 and r > ind["rsi_prev"] else "❌"
    macd_ok = "✅" if macd_hist > 0 and macd_hist > ind["macd_prev"] else "❌"
    stoch_ok = "✅" if ind["stoch_prev"] < 0.2 and stoch_k > ind["stoch_prev"] else "❌"

    all_long = bull_ok and rsi_ok == "✅" and macd_ok == "✅" and stoch_ok == "✅"
    signal_str = "⚡ LONG SIGNAL DETECTED!" if all_long else "   Waiting for signal..."

    position_str = "None"
    if position:
        pnl_distance = price - position["entry"] if position["side"] == "long" else position["entry"] - price
        position_str = (
            f"{position['side'].upper()} | entry={position['entry']} "
            f"| size={position['size']} | PnL dist={pnl_distance:+.2f}"
        )

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  HL-MOMENTUM-BOT  [{ts} UTC]  {cfg['symbol']}-PERP {cfg['timeframe']}          │
├─────────────────────────────────────────────────────────────┤
│  💰 Price     : {price:<14.4f}   VWAP  : {vwap_value:.4f}
│  📈 Trend     : {trend_str}
├─────────────────────────────────────────────────────────────┤
│  FIBONACCI CASCADE  (bull: 8 > 21 > 55 > 89 > 233)
│  {link1} Price > VWAP   : {price:.2f} {'>' if price > vwap_value else '<'} {vwap_value:.2f}
│  {link2} EMA8  > EMA21  : {ema8:.2f} {'>' if ema8 > ema21 else '<'} {ema21:.2f}
│  {link3} EMA21 > EMA55  : {ema21:.2f} {'>' if ema21 > ema55 else '<'} {ema55:.2f}
│  {link4} EMA55 > EMA89  : {ema55:.2f} {'>' if ema55 > ema89 else '<'} {ema89:.2f}
│  {link5} EMA89 > EMA233 : {ema89:.2f} {'>' if ema89 > ema233 else '<'} {ema233:.2f}
├─────────────────────────────────────────────────────────────┤
│  MOMENTUM CONFIRMATION
│  {rsi_ok} RSI(14)      : {r:.1f}  (prev: {ind['rsi_prev']:.1f})   [>45 and rising]
│  {macd_ok} MACD hist   : {macd_hist:.5f}  (prev: {ind['macd_prev']:.5f})  [>0 and rising]
│  {stoch_ok} StochRSI K : {stoch_k:.3f}  (prev: {ind['stoch_prev']:.3f})  [left <0.20]
│  📊 ATR(14)      : {a:.4f}
├─────────────────────────────────────────────────────────────┤
│  📦 Position  : {position_str}
│  {signal_str}
│  ⏱️  Next candle in : {next_candle_in:.0f}s
└─────────────────────────────────────────────────────────────┘""", flush=True)


class HLClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.symbol = cfg["symbol"]
        self.testnet = cfg.get("testnet", True)
        url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(url, skip_ws=True)
        wallet = eth_account.Account.from_key(cfg["secret_key"])
        self.exchange = Exchange(wallet, url, account_address=cfg["account_address"])
        log.info(f"HLClient initialized | {'TESTNET' if self.testnet else 'MAINNET'} | {self.symbol}")

    def get_candles(self, interval: str, limit: int = 600) -> Dict[str, np.ndarray]:
        now_ms = int(time.time() * 1000)
        tf_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        since = now_ms - limit * tf_seconds.get(interval, 900) * 1000
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
        o, h, l, c, v, t = [], [], [], [], [], []
        for candle in raw:
            o.append(float(candle["o"]))
            h.append(float(candle["h"]))
            l.append(float(candle["l"]))
            c.append(float(candle["c"]))
            v.append(float(candle["v"]))
            t.append(int(candle["t"]))
        return {
            "open": np.array(o, dtype=float),
            "high": np.array(h, dtype=float),
            "low": np.array(l, dtype=float),
            "close": np.array(c, dtype=float),
            "volume": np.array(v, dtype=float),
            "time": np.array(t),
        }

    def get_balance(self) -> float:
        return float(self.info.user_state(CFG["account_address"])["crossMarginSummary"]["accountValue"])

    def get_position(self) -> Optional[Dict[str, Any]]:
        for pos in self.info.user_state(CFG["account_address"]).get("assetPositions", []):
            if pos["position"]["coin"] == self.symbol:
                size = float(pos["position"]["szi"])
                if size != 0:
                    return {
                        "side": "long" if size > 0 else "short",
                        "size": abs(size),
                        "entry": float(pos["position"]["entryPx"]),
                    }
        return None

    def set_leverage(self, leverage: int):
        try:
            self.exchange.update_leverage(leverage, self.symbol, is_cross=True)
            log.info(f"Leverage set to: {leverage}x")
        except Exception as e:
            log.warning(f"Error setting leverage: {e}")

    def _mid_price(self) -> float:
        return float(self.info.all_mids()[self.symbol])

    def place_market(self, is_buy: bool, size: float):
        price = round(self._mid_price() * (1.0015 if is_buy else 0.9985), 2)
        result = self.exchange.order(
            self.symbol,
            is_buy,
            size,
            price,
            {"limit": {"tif": "Ioc"}},
            reduce_only=False,
        )
        log.info(f"{'BUY' if is_buy else 'SELL'} MARKET | size={size} | ~price={price} | {result}")
        return result

    def close_position(self, is_long: bool, size: float):
        price = round(self._mid_price() * (0.998 if is_long else 1.002), 2)
        result = self.exchange.order(
            self.symbol,
            not is_long,
            size,
            price,
            {"limit": {"tif": "Ioc"}},
            reduce_only=True,
        )
        log.info(f"CLOSE {'LONG' if is_long else 'SHORT'} | size={size} | ~price={price} | {result}")
        return result

    def coin_decimals(self) -> int:
        for asset in self.info.meta()["universe"]:
            if asset["name"] == self.symbol:
                return asset.get("szDecimals", 3)
        return 3


class RiskManager:
    def __init__(self, cfg):
        self.risk_pct = cfg["risk_pct"] / 100

    def position_size(self, balance, entry, stop_loss, decimals=3) -> float:
        distance = abs(entry - stop_loss) / entry
        if distance == 0:
            return 0.0
        factor = 10 ** decimals
        raw_size = (balance * self.risk_pct / distance / entry)
        return max(math.floor(raw_size * factor) / factor, 10 ** (-decimals))


class SignalEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def analyse(self, data) -> Dict[str, Any]:
        closes = data["close"]
        min_len = self.cfg.get("ema_trend", 233) + 50
        if len(closes) < min_len:
            return {"signal": 0, "reason": f"insufficient data ({len(closes)}/{min_len})"}

        ind = compute_indicators(data, self.cfg)

        if any(math.isnan(ind[k]) for k in ["rsi", "macd_hist", "stoch_k", "atr", "vwap"]):
            return {"signal": 0, "reason": "incomplete indicators", "ind": ind}

        bull_ok, bear_ok, _ = cascade_status(ind)
        atr_mult = self.cfg.get("atr_mult", 1.5)
        rr = self.cfg.get("rr", 2.0)

        long_ok = (
            bull_ok
            and ind["rsi"] > 45 and ind["rsi"] > ind["rsi_prev"]
            and ind["macd_hist"] > 0 and ind["macd_hist"] > ind["macd_prev"]
            and ind["stoch_prev"] < 0.2 and ind["stoch_k"] > ind["stoch_prev"]
        )
        if long_ok:
            risk = atr_mult * ind["atr"]
            return {
                "signal": 1,
                "side": "long",
                "sl": round(ind["price"] - risk, 2),
                "tp": round(ind["price"] + risk * rr, 2),
                "reason": "EMA8>21>55>89>233 + VWAP | RSI rising | MACD rising | StochRSI oversold->up",
                "ind": ind,
            }

        if self.cfg.get("enable_shorts", False):
            short_ok = (
                bear_ok
                and ind["rsi"] < 55 and ind["rsi"] < ind["rsi_prev"]
                and ind["macd_hist"] < 0 and ind["macd_hist"] < ind["macd_prev"]
                and ind["stoch_prev"] > 0.8 and ind["stoch_k"] < ind["stoch_prev"]
            )
            if short_ok:
                risk = atr_mult * ind["atr"]
                return {
                    "signal": 1,
                    "side": "short",
                    "sl": round(ind["price"] + risk, 2),
                    "tp": round(ind["price"] - risk * rr, 2),
                    "reason": "EMA8<21<55<89<233 + VWAP | RSI falling | MACD falling | StochRSI overbought->down",
                    "ind": ind,
                }

        return {"signal": 0, "reason": "no confluence", "ind": ind}


class MomentumBot:
    def __init__(self):
        self.cfg = CFG
        self.hl = HLClient(self.cfg)
        self.engine = SignalEngine(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.hl.set_leverage(self.cfg["leverage"])
        self.decimals = self.hl.coin_decimals()
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

    def run(self):
        log.info("Bot running... Press Ctrl+C to stop.")
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("Bot stopped by user.")
                break
            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(30)

    def _tick(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"── Tick {ts} ─────────────────────────────")
        data = self.hl.get_candles(self.cfg["timeframe"], self.cfg.get("lookback", 600))
        price = float(data["close"][-1])
        log.info(f"Current price : {price}")

        position = self.hl.get_position()
        if position:
            log.info(f"Open position: {position['side'].upper()} | size={position['size']} | entry={position['entry']}")
            self._manage_position(position, price)
            self._monitor_loop(data)
            return

        signal = self.engine.analyse(data)
        log.info(f"Signal : {signal['signal']} | {signal.get('side', '-')} | {signal.get('reason', '')}")

        if signal["signal"] == 1:
            balance = self.hl.get_balance()
            size = self.risk.position_size(balance, price, signal["sl"], self.decimals)
            log.info(f"Balance: {balance:.2f} | Size: {size} | SL: {signal['sl']} | TP: {signal['tp']}")
            if size > 0:
                self.hl.place_market(signal["side"] == "long", size)
                self.active_sl = signal["sl"]
                self.active_tp = signal["tp"]
            else:
                log.warning("Invalid size — order skipped.")

        self._monitor_loop(data)

    def _monitor_loop(self, data_snapshot):
        tf_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        period = tf_seconds.get(self.cfg["timeframe"], 900)
        now = time.time()
        candle_end = now + (period - (now % period)) + 2
        min_len = self.cfg.get("ema_trend", 233) + 50
        has_indicators = len(data_snapshot["close"]) >= min_len

        while True:
            remaining = candle_end - time.time()
            if remaining <= 0:
                break
            try:
                live_price = float(self.hl.info.all_mids()[self.cfg["symbol"]])
                data_snapshot["close"][-1] = live_price
            except Exception:
                pass

            if has_indicators:
                try:
                    indicators = compute_indicators(data_snapshot, self.cfg)
                    position = self.hl.get_position()
                    print_monitor(indicators, position, self.cfg, remaining)
                except Exception as e:
                    log.debug(f"Monitor error: {e}")
            else:
                price = float(data_snapshot["close"][-1])
                print(
                    f"\r  💰 {self.cfg['symbol']} = {price}  | Loading indicators... next candle in {remaining:.0f}s",
                    end="",
                    flush=True,
                )

            time.sleep(MONITOR_INTERVAL)

    def _manage_position(self, position, price):
        if self.active_sl is None or self.active_tp is None:
            log.warning("SL/TP not defined")
            return
        is_long = position["side"] == "long"
        hit_tp = (is_long and price >= self.active_tp) or (not is_long and price <= self.active_tp)
        hit_sl = (is_long and price <= self.active_sl) or (not is_long and price >= self.active_sl)
        if hit_tp:
            log.info(f"TAKE PROFIT @ {price} (TP={self.active_tp})")
            self.hl.close_position(is_long, position["size"])
            self._reset()
        elif hit_sl:
            log.info(f"STOP LOSS @ {price} (SL={self.active_sl})")
            self.hl.close_position(is_long, position["size"])
            self._reset()
        else:
            log.info(
                f"Position OK → dist_TP={abs(price - self.active_tp):.2f} | dist_SL={abs(price - self.active_sl):.2f}"
            )

    def _reset(self):
        self.active_sl = None
        self.active_tp = None


if __name__ == "__main__":
    bot = MomentumBot()
    bot.run()
