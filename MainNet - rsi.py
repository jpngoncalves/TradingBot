import websocket, json, pprint, talib, numpy
import config
from binance.client import Client
from binance.enums import *

client = Client(config.API_KEY, config.API_SECRET)

TRADE_SYMBOL = 'BTCUSDT'
TRADE_QUANTITY = 1

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

trade_socket_symbol = str(TRADE_SYMBOL).replace('/','')
SOCKET = "wss://stream.binance.com:9443/ws/%s@kline_5m" % trade_socket_symbol.lower() 
# The socket streams 1min candles, if you desire a different candle period, change kline_1m to kline_3m, kline_5m, or kline_15m

closes = []
in_position = True


def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
    try:
        print("Sending order...")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
        print(order, file=open('Logs/rsi.txt', 'a'))
    except Exception as e:
        print("An exception occured - {}".format(e))
        return False

    return True

def on_open(ws):
    print('Opened connection...')

def on_close(ws):
    print('Closed connection...')

def on_message(ws, message):
    global closes, in_position
    
    print('Received message...')
    json_message = json.loads(message)
    pprint.pprint(json_message)

    candle = json_message['k']

    is_candle_closed = candle['x']
    close = candle['c']

    if is_candle_closed:
        print("Candle closed at {}".format(close))
        closes.append(float(close))
        print("Closes")
        print(closes)

        if len(closes) > RSI_PERIOD:
            np_closes = numpy.array(closes)
            rsi = talib.RSI(np_closes, RSI_PERIOD)
            print("All RSIs calculated so far...")
            print(rsi)
            last_rsi = rsi[-1]
            print("The current RSI is {}".format(last_rsi))

            if last_rsi > RSI_OVERBOUGHT:
                if in_position:
                    print("Overbought! Sell! Sell! Sell!")
                    # put binance sell logic here
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = False
                else:
                    print("It is Overbought, but we don't own any. Nothing to do...")
            
            if last_rsi < RSI_OVERSOLD:
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