# HL-SMC-BOT — Hyperliquid Futures Trading Bot

## Estratégia
**Smart Money Concepts (SMC) + EMA Trend Filter**

### LONG quando:
1. EMA8 > EMA21 > EMA55 (tendência bullish confirmada)
2. PSAR abaixo do preço (momentum bullish)
3. Preço acima do VWAP (filtro institucional)
4. BOS/CHoCH bullish **ou** entrada em Order Block / FVG bullish

### SHORT quando:
1. EMA8 < EMA21 < EMA55 (tendência bearish confirmada)
2. PSAR acima do preço (momentum bearish)
3. Preço abaixo do VWAP (filtro institucional)
4. BOS/CHoCH bearish **ou** entrada em Order Block / FVG bearish

### Gestão de Risco
- **Stop Loss** : ATR × 1.5
- **Take Profit**: SL × 2.0 (R:R 1:2)
- **Tamanho**   : % do account em risco (configurável)

---

## Como ligar à API Hyperliquid

1. Vai a https://app.hyperliquid.xyz/API
2. Gera uma **API Wallet** e guarda a private key (`0x...`)
3. No `config.json`:
   - `account_address` = endereço público da tua wallet principal
   - `secret_key`      = private key da API Wallet
   - `testnet`         = true para testar, false para mainnet

---

## Estrutura de ficheiros

```
DEV/hyperliquid-smc-bot/
├── bot.py           # Bot principal
├── config.json      # Configuração específica deste bot
├── requirements.txt
├── estrategia.txt   # Descrição em texto da estratégia
└── logs/
    └── bot.log
```
