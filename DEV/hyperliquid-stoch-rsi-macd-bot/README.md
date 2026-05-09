# HL-STOCH-RSI-MACD-BOT

Bot de trading de futuros perpétuos na Hyperliquid baseado em:

- Stochastic Slow (14,3,3)
- RSI(14)
- MACD(12,26,9)
- Filtro de tendência com EMAs 8/21/55
- Filtro institucional com VWAP

A lógica é inspirada no vídeo original para Binance, mas com:

- Adaptação à API oficial Hyperliquid
- Gestão de risco em % do account
- SL/TP fixos por multiplicador (ex.: -1% / +2%)
- Código em numpy puro (sem TA-Lib/ta)

Ver `estrategia.txt` para descrição completa das regras.
