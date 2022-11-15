import config
import requests 
import numpy as np 
import talib 
import time
from binance.client import Client 

client = Client(config.API_KEY, config.API_SECRET)  

TRADE_SYMBOL = "BTCUSDT"  
TRADE_TIME_PERIOD= "15m"  
TRADE_LIMIT = "100"  #Original = 100  
TRADE_QUANTITY = 100 
TRADE_EMA_ST = 13    #Original = 8
TRADE_EMA_LT = 144   #Original = 21


def place_order(order_type):
         
    if(order_type == "buy"):
        
        order = client.create_order(symbol=TRADE_SYMBOL, side="buy",quantity=TRADE_QUANTITY,type="MARKET")
        time.sleep(60)  
    else:
        order = client.create_order(symbol=TRADE_SYMBOL, side="sell", quantity=TRADE_QUANTITY,type="MARKET")
        time.sleep(60)  

    print("Order Placed Successfully!!!") 
    print(order)
    print(order, file=open('Logs/ema-crossover.txt', 'a'))
    
    return

def get_data():
    url = "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit={}".format(TRADE_SYMBOL, TRADE_TIME_PERIOD, TRADE_LIMIT)
    res = requests.get(url) 
     
    return_data = []
    for each in res.json():
        return_data.append(float(each[4]))
    return np.array(return_data)

#define main entry point for the script 
def main():
    buy = False #it means we are yet to buy and have not bought
    sell = True #we have not sold , but if you want to buy first then set it to True
    ema_ST = None  
    ema_LT = None  

    #we also need to store the last variables that was the value for the ema_8 and ema_21, so we can compare
    last_ema_ST = None 
    last_ema_LT = None 

    print("Started Running...")
     
    while True:
        closing_data = get_data()
        last_closing_data = closing_data[-1]
         
        ema_ST = talib.EMA(closing_data,TRADE_EMA_ST)[-1]  
        ema_LT = talib.EMA(closing_data, TRADE_EMA_LT)[-1]  

        Diff = ema_ST - ema_LT

        #Data info... 
        print("----------------------------------------------------")
        print("******* " + TRADE_SYMBOL + " *******")
        print("Last Close |", last_closing_data)
        print("Short Term EMA |", ema_ST)     
        print("Long Term EMA |",ema_LT)
        print("----------------------------------------------------") 
        print("******* EMA Diference... *******")
        print("Short Term EMA - Long Term EMA | ", (Diff))   
        time.sleep(1)

        if(ema_ST > ema_LT and last_ema_ST):  
            if(last_ema_ST < last_ema_LT and not buy):  
                print("Buy! Buy! Buy!")
                place_order("buy")                
                buy = True 
                sell = False #switch the values for next order

        if(ema_LT > ema_ST and last_ema_LT):   
            if(last_ema_LT < last_ema_ST and not sell): 
                print("Sell! Sell! Sell!")
                place_order("sell")                
                sell = True 
                buy = False #switching values for next order 

        #at last we are setting the current values as last one 
        last_ema_ST = ema_ST 
        last_ema_LT = ema_LT
        #return
        
if __name__ == "__main__":
    main()