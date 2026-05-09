HL-MOMENTUM-BOT — Hyperliquid Futures Trend + Momentum Bot

## Descrição

Bot de Futuros Perpétuos na Hyperliquid que segue a tendência em 15m com
entradas em pullbacks bem definidos.

- Timeframe: **15m** (config.json)
- Mercado: perps (por defeito **BTC-PERP**)
- Indicadores:
  - EMA(21/55/233) → tendência de curto/médio/longo
  - VWAP → referência institucional de preço justo
  - RSI(14) → força do movimento
  - MACD(12,26,9) (histograma) → momentum
  - StochRSI(14,3,3) → timing de saída de oversold/overbought
  - ATR(14) → dimensão de SL/TP

## Regras principais

### LONG
1. Tendência bullish:
   - Preço > VWAP
   - EMA21 > EMA55 > EMA233
2. Pullback saudável:
   - RSI > 45 e RSI actual > RSI anterior
   - MACD hist > 0 e a melhorar
   - StochRSI: K_prev < 0.2 e K_actual > K_prev (sair de oversold)
3. SL/TP:
   - SL = entry − ATR × atr_mult
   - TP = entry + ATR × atr_mult × rr

### SHORT (opcional)
- Activado com `enable_shorts = true` em config.json.
- Condições simétricas em tendência bearish.

## Gestão de risco

- `risk_pct`: percentagem do account arriscada por trade entre entry e SL.
- Tamanho da posição calculado automaticamente em função da distância ao SL.
- `atr_mult` e `rr` controlam a agressividade (ruído vs. alvo de lucro).

## Uso

1. Preenche `config.json` com:
   - `account_address` e `secret_key` da tua conta/ carteira de trading HL.
   - `testnet = true` enquanto testas.
2. Instala dependências:

```bash
pip install -r DEV/hyperliquid-momentum-bot/requirements.txt
```

3. Corre o bot:

```bash
python DEV/hyperliquid-momentum-bot/bot.py
```
