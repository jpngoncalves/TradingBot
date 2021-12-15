import websocket, json, pprint, talib, numpy
import config
from binance.client import Client
from binance.enums import *
import pandas as pd

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m" #Change manually Symbol and Time Period

TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 10000

STOCH_RSI_PERIOD = 14
STOCH_RSI_OVERBOUGHT = 70
STOCH_RSI_OVERSOLD = 30

closes = []
in_position = False

client = Client(config.API_KEY, config.API_SECRET)

def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
    try:
        print("Sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("An exception occured - {}".format(e))
        return False

    return True

def on_open(ws):
    print('Opened connection')

def on_close(ws):
    print('Closed connection')

def on_message(ws, message):
    global closes, in_position
    
    #print('Received message...')
    json_message = json.loads(message)
    #pprint.pprint(json_message)

    candle = json_message['k']

    is_candle_closed = candle['x']
    close = candle['c']

    if is_candle_closed:
        print("Candle closed at {}".format(close))
        closes.append(float(close))
        print("Closes")
        print(closes)

        if len(closes) > STOCH_RSI_PERIOD:
            np_closes = numpy.array(closes)
            stoch_rsi = talib.STOCHRSI(np_closes, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
            print("All STOCH RSIs calculated so far...")
            print(stoch_rsi)
            last_stoch_rsi = stoch_rsi[-1]
            print("The current STOCH RSI is {}".format(last_stoch_rsi))

            if last_stoch_rsi > STOCH_RSI_OVERBOUGHT:
                if in_position:
                    print("Overbought! Sell! Sell! Sell!")
                    # put binance sell logic here
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = False
                else:
                    print("It is Overbought, but we don't own any. Nothing to do...")
            
            if last_stoch_rsi < STOCH_RSI_OVERSOLD:
                if in_position:
                    print("It is Oversold, but you already own it, nothing to do...")
                else:
                    print("Oversold! Buy! Buy! Buy!")
                    # put binance buy order logic here
                    order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = True
                
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()