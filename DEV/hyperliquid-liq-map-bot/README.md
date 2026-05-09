HL-LIQ-MAP-BOT — Hyperliquid + CoinGlass Liquidity Map Bot

## Ideia geral
Bot de Futuros Perpétuos na Hyperliquid que tenta alinhar entradas e saídas
com o **mapa de liquidações** da CoinGlass, tal como fazem muitos traders
quantitativos/institucionais:

- Timeframe base: **15m** (configurável)
- Mercado: perps (por defeito **BTC-PERP** na Hyperliquid)
- Edge principal: evitar entrar **directamente em cima de clusters de
  liquidações na direcção errada** e procurar assimetrias onde existam
  **mais shorts acima** (para LONG) ou **mais longs abaixo** (para SHORT).

O bot combina três pilares:
1. **Tendência institucional** (EMA 21/55/233 + VWAP)
2. **Osciladores/momentum** (RSI, MACD, StochRSI)
3. **Mapa de liquidações CoinGlass** (heatmap / clusters agregados).

---

## Estratégia

### 1. Filtro de tendência (institucional)

- Calcula EMAs em 15m:
  - `ema_fast = EMA(21)`
  - `ema_slow = EMA(55)`
  - `ema_trend = EMA(233)`
- Calcula VWAP usando high/low/close/volume.

**Tendência Bullish:**
- `Preço > VWAP`
- `ema_fast > ema_slow > ema_trend`

**Tendência Bearish:**
- `Preço < VWAP`
- `ema_fast < ema_slow < ema_trend`

O bot só considera LONGs em contexto bullish e SHORTs (se activado) em
contexto bearish.

### 2. Osciladores / Momentum

Calcula indicadores clássicos:

- `RSI(14)`
- `MACD(12,26,9)` → usa o **histograma** (macd_line - signal)
- `StochRSI(14,3,3)` → normalizado [0,1]
- `ATR(14)` → para dimensionar SL/TP

Pullback "saudável" para LONG:
- RSI > 45 e a subir face à vela anterior.
- MACD hist > 0 e **a melhorar** (hist actual > hist anterior).
- StochRSI a sair de oversold: `K_prev < 0.2` e `K_actual > K_prev`.

Pullback "saudável" para SHORT:
- RSI < 55 e a descer.
- MACD hist < 0 e a piorar.
- StochRSI a sair de overbought: `K_prev > 0.8` e `K_actual < K_prev`.

Na versão inicial, SHORTs estão desactivados (`enable_shorts = false`) para
manter o risco controlado; podem ser activados no `config.json`.

### 3. Mapa de Liquidações (CoinGlass)

O módulo CoinGlass faz:
- Chamada ao endpoint v4 de heatmap de liquidações:
  - `/api/futures/liquidation/heatmap/model2` (ver documentação CoinGlass API v4)
- Filtra os níveis de preço num intervalo `±liq_window_pct` (ex: ±5%) à volta
  do preço actual.
- Para cada nível, soma liquidações long e short:
  - `long_above`: liquidações long **acima** do preço actual
  - `short_above`: liquidações short **acima**
  - `long_below`: liquidações long **abaixo**
  - `short_below`: liquidações short **abaixo**
- Calcula métricas de assimetria:
  - `net_short_above = short_above - long_above`
  - `net_long_below  = long_below - short_below`

Interpretação simplificada:
- Para **LONG** queremos que `net_short_above > 0` ⇒ há mais shorts pendurados
  acima que podem ser liquidados num squeeze.
- Para **SHORT** queremos que `net_long_below > 0` ⇒ há mais longs pendurados
  abaixo.

Se a CoinGlass API não estiver configurada (sem API key ou erro), o bot
continua a funcionar APENAS com sinais técnicos (EMA/VWAP/RSI/MACD/StochRSI).

---

## Regras de Entrada

### LONG

Condições necessárias:
1. **Tendência bullish**:
   - Preço > VWAP
   - `EMA21 > EMA55 > EMA233`
2. **Pullback saudável**:
   - RSI > 45 e RSI actual > RSI anterior
   - MACD hist > 0 e hist actual > hist anterior
   - StochRSI: `K_prev < 0.2` e `K_actual > K_prev`
3. **Mapa de Liquidações (se disponível)**:
   - `net_short_above > 0` ⇒ há mais shorts acima (potencial squeeze)

Quando tudo alinha, o bot calcula SL/TP com base em ATR e envia ordem de
mercado LONG com tamanho calculado pelo RiskManager.

### SHORT (opcional)

Se `enable_shorts=true` em `config.json`, o bot também pode shortar:
1. **Tendência bearish**:
   - Preço < VWAP
   - `EMA21 < EMA55 < EMA233`
2. **Pullback saudável**:
   - RSI < 55 e RSI actual < RSI anterior
   - MACD hist < 0 e a piorar
   - StochRSI: `K_prev > 0.8` e `K_actual < K_prev`
3. **Mapa de Liquidações**:
   - `net_long_below > 0` ⇒ há mais longs abaixo (potencial cascata).

---

## Gestão de Risco

- `ATR_MULT` (`atr_mult` no config):
  - `risk = ATR(14) × atr_mult`
- SL e TP:
  - Para LONG: `SL = entry − risk`, `TP = entry + risk × rr`
  - Para SHORT: `SL = entry + risk`, `TP = entry − risk × rr`
- `rr` (risk-reward):
  - Por defeito 2.0 (R:R ≈ 1:2)
- Tamanho da posição:
  - Calculado para arriscar `risk_pct`% do account entre entry e SL.

---

## Configuração

Ver ficheiro `config.json`:

```json
{
  "account_address": "0xSEU_ENDERECO_PUBLICO_AQUI",
  "secret_key": "0xSUA_CHAVE_PRIVADA_API_AQUI",
  "testnet": true,
  "symbol": "BTC",
  "timeframe": "15m",
  "leverage": 5,
  "risk_pct": 1.0,
  "ema_fast": 21,
  "ema_slow": 55,
  "ema_trend": 233,
  "rsi_length": 14,
  "atr_mult": 1.5,
  "rr": 2.0,
  "enable_shorts": false,
  "lookback": 400,
  "liq_window_pct": 0.05,
  "coinglass_base_url": "https://open-api-v4.coinglass.com",
  "coinglass_symbol": "BTC",
  "coinglass_api_key": "INSERE_AQUI_O_TEU_API_KEY_OU USA VARIÁVEL DE AMBIENTE COINGLASS_API_KEY",
  "log_level": "INFO"
}
```

- Preenche `account_address` e `secret_key` com a tua conta Hyperliquid ou
  API wallet.
- Garante que estás em `testnet=true` enquanto experimentas.
- Para activar CoinGlass de forma segura, define a variável de ambiente
  `COINGLASS_API_KEY` no servidor:

```bash
export COINGLASS_API_KEY="A_TUA_CHAVE_AQUI"
```

ou coloca directamente no `config.json` (menos seguro).

---

## Requisitos

Ver `requirements.txt`:

- `hyperliquid-python-sdk`
- `eth-account`
- `numpy`
- `requests`

Instalação:

```bash
pip install -r DEV/hyperliquid-liq-map-bot/requirements.txt
python DEV/hyperliquid-liq-map-bot/bot.py
```

---

## Roadmap / Melhorias futuras

- Backtests históricos com dados BTC perp + endpoints CoinGlass para
  validar efectivamente a edge de liquidações.
- Adicionar módulo de gestão de exposição por regime de funding/open
  interest (CoinGlass também fornece estes dados).
- Short side activado por defeito assim que estiveres confortável com a
  lógica long.
