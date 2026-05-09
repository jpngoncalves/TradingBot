HL-EMA-RSI-MACD-BOT — Hyperliquid Futures Trading Bot

## Estratégia
Versão adaptada do bot EMA/RSI/MACD (2022) para Futuros Perpétuos na Hyperliquid.

- Timeframe: **15m**
- Mercado: perps (BTC-PERP por defeito)
- Indicadores:
  - EMA curta 13, EMA longa 144
  - RSI(14)
  - MACD(12,26,9) (histograma)
  - StochRSI(14,3,3)
  - ATR(14) para SL/TP

### Entradas LONG
1. Trigger: EMA13 < EMA144 em pelo menos 1 das últimas `lags` velas.
2. EMA13 < EMA144 na vela actual.
3. RSI(14) < 30 (oversold).
4. MACD hist < 0 mas a melhorar (hist actual > hist anterior).
5. StochRSI a sair de oversold (K_prev < 0.2 e K_actual > K_prev).

### Saídas
- SL baseado em ATR: `SL = entry − ATR × atr_mult`.
- TP com relação R:R = `rr` (por defeito 1:2).
- Quando preço toca SL ou TP, o bot envia ordem de fecho reduce-only.

### Gestão de Risco
- Tamanho da posição calculado para arriscar `risk_pct`% do valor da conta por trade.
- Leverage configurável (`leverage`, por defeito 5x, cross).

Ver `estrategia.txt` para descrição detalhada.
