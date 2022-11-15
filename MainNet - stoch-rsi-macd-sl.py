# Trading bot https://youtu.be/X50-c54BWV8
# Strats: Stochastics Slow, RSI, MACD, Target Profit and Stop loss - https://www.youtube.com/watch?v=hh3BKTFE1dc&t=0s

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import ta
from time import sleep
import config
import numpy as np

client = Client(config.API_KEY, config.API_SECRET)

TRADE_SYMBOL = 'BTCUSDT'
TRADE_ASSET = 'BTC'
TRADE_PAIR = 'USDT'
TRADE_QUANTITY = 100
POSITION = False

TRADE_TIME_PERIOD= '5m'  
TRADE_LIMIT = '100'  

TRADE_SL = 0.985 # 0.995 = 0.5% stop loss
TRADE_TARGET = 1.02 # 1.005 = 0.5% take profit 

TRADE_RSI = 14
TRADE_STOCH_WINDOW = 14
TRADE_STOCH_SMOTH_WINDOW = 3

def get_minute_data(pair, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(pair, interval, lookback + 'hour ago UTC'))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame

def applytechnicals(df):
    # Stochastic values
    df['%K'] = ta.momentum.stoch(df.High,df.Low, df.Close, window=TRADE_STOCH_WINDOW, smooth_window=TRADE_STOCH_SMOTH_WINDOW)
    df['%D'] = df['%K'].rolling(TRADE_STOCH_SMOTH_WINDOW).mean()
    
    # Relative strength index
    df['rsi'] = ta.momentum.rsi(df.Close, window=TRADE_RSI)
    
    # Moving average convergence/divergence
    df['macd'] = ta.trend.macd_diff(df.Close) 
    
    df.dropna(inplace=True)
    return df

class Signals:
    def __init__(self, df, lags):
        self.df = df
        self.lags = lags
    
    def get_trigger(self):
        dfx = pd.DataFrame()
        for i in range(self.lags + 1):
            # Check if %K, %D cross under 20
            mask = (self.df['%K'].shift(i) < 20) & (self.df['%D'].shift(i) < 20)
            dfxn = pd.DataFrame([mask])
            dfx = pd.concat([dfx, dfxn])
        return dfx.sum(axis=0)
    
    def decide(self):
        self.df['Trigger'] = np.where(self.get_trigger(), 1, 0)
        self.df['Buy'] = np.where(
            (self.df.Trigger) & 
            (self.df['%K'].between(20, 80)) & (self.df['%D'].between(20, 80)) &
            (self.df.rsi > 50) &
            (self.df.macd > 0),
            1, 0
        )
        return self.df

def strat(pair, qty, open_position=False):
    mindata = get_minute_data(pair, TRADE_TIME_PERIOD, TRADE_LIMIT)
    techdata = applytechnicals(mindata)
    inst = Signals(techdata, 25) #original = 25 #lags very important parameter 2, 3 or 5
    data = inst.decide()
    print(data)
    #print('--------------------------------')
    #print(f'Current Close is '+str(mindata.Close.iloc[-1]))
    print('--------------------------------')
    sleep(2)
    if data.Buy.iloc[-1]:
        
        try:
            buyorder = client.create_order(
                symbol=pair,
                side='BUY',
                type='MARKET',
                quantity= qty
            )
            print(buyorder)
            print(buyorder, file=open('Logs/stoch-rsi-macd-sl.txt', 'a'))
            buyprice = float(buyorder['fills'][0]['price'])
            print('Buy at price: {}, stop: {}, target price: {}'.format(buyprice, buyprice * TRADE_SL, buyprice * TRADE_TARGET))
#            print(get_main_free_balances())
            open_position = True
        except BinanceAPIException as e:
            print('Error: {} ({})'.format(e.message, e.status_code))
            sleep(5)

    while open_position:
        sleep(0.5)
        mindata = get_minute_data(pair, '1m', '2')
        if mindata.Close[-1] <= buyprice * TRADE_SL or mindata.Close[-1] >= buyprice * TRADE_TARGET:
            
            sellorder = client.create_order(
                symbol=pair,
                side='SELL',
                type='MARKET',
                quantity= qty
            )
            print(sellorder)
            print(sellorder, file=open('Logs/stoch-rsi-macd-sl.txt', 'a'))
            print('Sell at stop: {}, target: {}'.format(buyprice * TRADE_SL, buyprice * TRADE_TARGET))
#            print(get_main_free_balances())
            print('Win/loss: {}%'.format(round((float(sellorder['fills'][0]['price']) / buyprice - 1) * 100, 3)))
            open_position = False
            sleep(5)
            break

#def clean_order(order):
#    relev_info = {
#        'OrderId':order['clientOrderId'],
#        'Time':pd.to_datetime(order['transactTime'], unit='ms'),
#        'Side':order['side'],
#        'Qty':float(order['fills'][0]['qty']),
#        'Commission':float(order['fills'][0]['commission']),
#        'Price':float(order['fills'][0]['price'])
#    }
#    df = pd.DataFrame([relev_info])
#    return df

#def get_main_balances():
#    for item in client.get_account()['balances']:
#        if item['asset'] == TRADE_ASSET:
#            print('ASSET:\tFree: {}, Locked: {}'.format(item['free'], item['locked']))
#        elif item['asset'] == TRADE_PAIR:
#            print('PAIR:\tFree: {}, Locked: {}'.format(item['free'], item['locked']))

#def get_main_free_balances():
#    asset = 0
#    pair = 0
#    for item in client.get_account()['balances']:
#        if item['asset'] == TRADE_ASSET:
#            asset = item['free']
#        elif item['asset'] == TRADE_PAIR:
#            pair = item['free']
#    return 'Free ASSET: {}, PAIR: {}'.format(asset, pair)

def main(args=None):
#    print(get_main_free_balances())

    while True:
        sleep(0.5)
        strat(TRADE_SYMBOL, TRADE_QUANTITY)

if __name__ == '__main__':
    print('Running...') 
    main()
