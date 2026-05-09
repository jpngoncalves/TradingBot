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

### Passo 1 — Gerar API Key
1. Vai a [https://app.hyperliquid.xyz/API](https://app.hyperliquid.xyz/API)
2. Clica **"Generate API Key"**
3. Autoriza a chave com a tua wallet (MetaMask/WalletConnect)
4. Guarda a **Private Key da API wallet** (começa com `0x`)

### Passo 2 — Editar config.json
```json
{
  "account_address": "0xSEU_ENDERECO_PUBLICO_PRINCIPAL",
  "secret_key":      "0xPRIVATE_KEY_DA_API_WALLET",
  "testnet": true
}
```
> ⚠️ NUNCA coloques a private key da tua wallet principal!  
> A `secret_key` deve ser da **API Wallet** gerada acima.  
> A `account_address` é o endereço público da **wallet principal**.

### Passo 3 — Instalar dependências
```bash
pip install -r requirements.txt
```

### Passo 4 — Testar em Testnet
```bash
# Garante que testnet: true no config.json
python bot.py
```
Faucet testnet: https://app.hyperliquid-testnet.xyz/

### Passo 5 — Mainnet
```json
"testnet": false
```

---

## Estrutura de ficheiros
```
DEV/
├── bot.py           # Bot principal
├── config.json      # Configuração (NÃO commitar com chaves reais!)
├── requirements.txt
├── .gitignore
├── logs/
│   └── bot.log
└── README.md
```

---

## Parâmetros configuráveis

| Parâmetro      | Default | Descrição                          |
|----------------|---------|------------------------------------|
| symbol         | BTC     | Activo a negociar                  |
| timeframe      | 15m     | Timeframe das velas                |
| leverage       | 5       | Alavancagem (cross margin)         |
| risk_pct       | 1.0     | % do account em risco por trade    |
| ema_fast       | 8       | EMA rápida                         |
| ema_slow       | 21      | EMA lenta                          |
| ema_trend      | 55      | EMA de tendência                   |
| ema_macro      | 233     | EMA macro (Fibonacci)              |
| swing_lookback | 50      | Lookback para swing points         |

---

## Auditoria de segurança ✅
- Sem chaves hardcoded — tudo em config.json
- config.json no .gitignore
- Testnet mode por defeito
- Stop Loss em todas as posições
- Logs em ficheiro + stdout
- Reconexão automática em erros
- Slippage control nas ordens de mercado
