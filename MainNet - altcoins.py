
import config
from binance.client import Client
import pandas as pd
import time

client = Client(config.API_KEY, config.API_SECRET)

TRADE_PAR = 'ADA'
TRADE_AMOUNT = 100
TRADE_SL = 0.985 #Original SL: 0.985
TRADE_TARGET = 1.02 #Original Target: 1.02

TRADE_TIME_PERIOD= '1m' #Enter kline value (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1D, 3D, 1W, 1M) 
TRADE_LIMIT = '100' 

balance = client.get_asset_balance(asset=TRADE_PAR)
print('Total Balance |', TRADE_PAR, balance['free'])

###################### | DATA TEST | ######################

#x = pd.DataFrame(client.get_ticker()) 
#print(x)

#y = x[x.symbol.str.contains(TRADE_PAR)] 
#print(y)

#z = y[~((y.symbol.str.contains('UP')) | (y.symbol.str.contains('DOWN')))]
#print(z)

#z1 = z[z.priceChangePercent == z.priceChangePercent.max()]
#print(z1)

###########################################################

#VARIABLES:
    #priceChangePercent
    #lastQty
    #bidQty
    #askQty
    #quoteVolume
    #volume

###########################################################

def get_top_symbol():
    all_pairs = pd.DataFrame(client.get_ticker())
    relev = all_pairs[all_pairs.symbol.str.contains(TRADE_PAR)]
    non_lev = relev[~((relev.symbol.str.contains('UP')) | (relev.symbol.str.contains('DOWN')))]
    top_symbol = non_lev[non_lev.priceChangePercent == non_lev.priceChangePercent.max()] #Change variables...
    top_symbol = top_symbol.symbol.values[0]
    return top_symbol
print(get_top_symbol())

def getminutedata(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback+' min ago UTC'))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame

def strategy(buy_amt, SL=TRADE_SL, Target=TRADE_TARGET, open_position=False):
    try:
        asset = get_top_symbol()
        df = getminutedata(asset, TRADE_TIME_PERIOD, TRADE_LIMIT)    
    except:
        time.sleep(61)
        asset: get_top_symbol()    
        df = getminutedata(asset, TRADE_TIME_PERIOD, TRADE_LIMIT)
    qty = round(buy_amt/df.Close.iloc[-1])
    if ((df.Close.pct_change() + 1).cumprod()).iloc[-1] > 1:
        order = client.create_order(symbol=asset, side='BUY', type='MARKET', quantity = qty)
        print(order)
        print(order, file=open('Logs/altcoins.txt', 'a'))
        buyprice = float(order['fills'][0]['price'])
        open_position = True
        while open_position: 
            try:
                df = getminutedata(asset, '1m', '2')
            except:
                print('Something wrent wrong. Script continues in 1 minute...')
                time.sleep(61)
                df = getminutedata(asset, '1m', '2')
            print('*************************')
            print('WORKING IN PROFITS...')
            print(f'Buyprice | ' + str(buyprice))
            print(f'Current Close | ' + str(df.Close[-1]))
            print(f'Target | ' + str(buyprice * Target))
            print(f'Stop Loss | ' + str(buyprice * SL))
            if df.Close[-1] <= buyprice * SL or df.Close[-1] >= buyprice * Target:
                order = client.create_order(symbol=asset, side='SELL', type='MARKET', quantity = qty)
                print(order)
                print(order, file=open('Logs/altcoins.txt', 'a'))
                break
while True:
    strategy(TRADE_AMOUNT)
    time.sleep(5)
    