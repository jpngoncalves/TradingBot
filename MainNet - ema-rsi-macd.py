# Strategy: EMA, RSI, MACD
# Backtest : 1min | ST EMA = 13 | LT EMA = 144 

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import ta
import talib
from time import sleep
import config
import numpy as np

client = Client(config.API_KEY, config.API_SECRET)

TRADE_SYMBOL = 'MATICBUSD'
TRADE_ASSET = 'MATIC'
TRADE_PAIR = 'BUSD'
TRADE_QUANTITY = 5000
IN_POSITION = False # Change 

TRADE_TIME_PERIOD= '15m'  
TRADE_LIMIT = '100'
TRADE_KLINE = '15min ago UTC' # Original 'min ago UTC'  
#EMA constants
TRADE_EMA_ST = 13    #Original : 8
TRADE_EMA_LT = 144    #Original : 21
#RSI constants
TRADE_RSI = 14
RSI_OVERBOUGHT = 70   #Original : 70
RSI_OVERSOLD = 30   #Original : 30
MACD = 0 # MACD > 0
#LAYS constants
TRADE_LAGS = 5 #original = 25 #lags very important parameter 2, 3 or 5

def get_minute_data(pair, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(pair, interval, lookback + TRADE_KLINE))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame

def applytechnicals(df):
    # EMA
    df['ema_ST'] = talib.EMA(df.Close, TRADE_EMA_ST) 
    df['ema_LT'] = talib.EMA(df.Close, TRADE_EMA_LT)
      
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
            mask = (self.df['ema_ST'].shift(i) < self.df['ema_LT'].shift(i))
            dfxn = pd.DataFrame([mask])
            dfx = pd.concat([dfx, dfxn])
            
        return dfx.sum(axis=0)
    
    def decide(self):
        self.df['Trigger'] = np.where(self.get_trigger(), 1, 0)
        self.df['Buy'] = np.where(
            (self.df.Trigger) & 
            (self.df.rsi < RSI_OVERSOLD) &
            (self.df.macd < 0) & 
            (self.df['ema_ST'] < self.df['ema_LT']), 
            1, 0
        )
        self.df['Sell'] = np.where(
            (self.df.Trigger) & 
            (self.df.rsi > RSI_OVERBOUGHT) &
            (self.df.macd > MACD) & 
            (self.df['ema_LT'] < self.df['ema_ST']), 
            1, 0
        )
        return self.df

def strat(pair, qty, open_position=IN_POSITION):
    mindata = get_minute_data(pair, TRADE_TIME_PERIOD, TRADE_LIMIT)
    techdata = applytechnicals(mindata)
    inst = Signals(techdata, TRADE_LAGS) 
    data = inst.decide()
    print(data)
    print('--------------------------')
    print('In Buying position...')
    print('--------------------------')
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
            print(buyorder, file=open('Logs/ema-rsi-macd.txt', 'a'))
            open_position = True
        except BinanceAPIException as e:
            print('Error: {} ({})'.format(e.message, e.status_code))
            sleep(15)

    while open_position:
        sleep(0.5)
        mindata = get_minute_data(pair, TRADE_TIME_PERIOD, TRADE_LIMIT)
        techdata = applytechnicals(mindata)
        inst = Signals(techdata, TRADE_LAGS) 
        data = inst.decide()
        print(data)
        print('--------------------------')
        print('In Selling position...')
        print('--------------------------')
        sleep(2)

        if data.Sell.iloc[-1]:
            sellorder = client.create_order(
                symbol=pair,
                side='SELL',
                type='MARKET',
                quantity= qty
            )
            print(sellorder)
            print(sellorder, file=open('Logs/ema-rsi-macd.txt', 'a'))
            open_position = False
            sleep(15)
            break

def main(args=None):

    while True:
        strat(TRADE_SYMBOL, TRADE_QUANTITY)
        sleep(5)

if __name__ == '__main__':
    print('Running...') 
    main()
