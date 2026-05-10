# Hyperliquid Momentum Bot

A momentum and trend-following futures bot for Hyperliquid perpetuals.

## Strategy

This bot trades with a full Fibonacci EMA cascade:
- EMA 8 → short-term momentum
- EMA 21 → short/medium trend
- EMA 55 → medium trend
- EMA 89 → medium/long trend
- EMA 233 → major trend

A long signal requires:
- Price above VWAP
- EMA 8 > EMA 21 > EMA 55 > EMA 89 > EMA 233
- RSI(14) above 45 and rising
- MACD histogram above 0 and rising
- StochRSI leaving oversold conditions

An optional short mode mirrors these rules in the opposite direction.

## Risk Management

- Position sizing based on account risk percentage
- ATR-based stop-loss
- Risk/reward take-profit target
- Leverage configurable in `config.json`

## Files

- `bot.py` — main bot logic
- `config.json` — bot configuration
- `strategy.txt` — strategy description
- `requirements.txt` — dependencies

## Run

```bash
pip install -r requirements.txt
python bot.py
```

## Notes

- Testnet is enabled by default
- Update your account address and private key in `config.json`
- Logs are written to `logs/bot.log`
