import config
import requests 
import numpy as np 
import talib 
import time
from binance.client  import Client 

client = Client(config.API_KEY, config.API_SECRET)  

TRADE_SYMBOL = "BTCUSDT"  
TRADE_TIME_PERIOD= "5m"  
TRADE_LIMIT = "200"  
TRADE_QUANTITY = 1 
TRADE_SMA_ST = 6
TRADE_SMA_LT = 14

# Tested - 5 minutes timeframe

def place_order(order_type):
         
    if(order_type == "buy"):
        
        order = client.create_order(symbol=TRADE_SYMBOL, side="buy",quantity=TRADE_QUANTITY,type="MARKET")
        time.sleep(60)  
    else:
        order = client.create_order(symbol=TRADE_SYMBOL, side="sell", quantity=TRADE_QUANTITY,type="MARKET")
        time.sleep(60)  

    print("Order Placed Successfully!!!") 
    print(order)
    print(order, file=open('Logs/sma-crossover.txt', 'a'))
    
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
    sma_ST = None  
    sma_LT = None  

    #we also need to store the last variables that was the value for the ema_8 and ema_21, so we can compare
    last_sma_ST = None 
    last_sma_LT = None 

    print("Started Running...")
     
    while True:
        closing_data = get_data()
        last_closing_data = closing_data[-1]
          
        sma_ST = talib.SMA(closing_data,TRADE_SMA_ST)[-1]  
        sma_LT = talib.SMA(closing_data, TRADE_SMA_LT)[-1]  

        Diff = sma_ST - sma_LT

        #Data info... 
        print("----------------------------------------------------")
        print("*******" + TRADE_SYMBOL + " *******")
        print("Last Close |", last_closing_data)
        print("Short Term SMA |", sma_ST)     
        print("Long Term SMA |",sma_LT)
        print("----------------------------------------------------") 
        print("******* SMA Diference... *******")
        print("Short Term SMA - Long Term SMA | ", (Diff)) 
        time.sleep(1)  
        
        if(sma_ST > sma_LT and last_sma_ST):  
            if(last_sma_ST < last_sma_LT and not buy):  
                print("Buy! Buy! Buy!")
                place_order("buy")                
                buy = True 
                sell = False #switch the values for next order

        if(sma_LT > sma_ST and last_sma_LT):   
            if(last_sma_LT < last_sma_ST and not sell): 
                print("Sell! Sell! Sell!")
                place_order("sell")                
                sell = True 
                buy = False #switching values for next order 

        #at last we are setting the current values as last one 
        last_sma_ST = sma_ST 
        last_sma_LT = sma_LT
        #return
        
if __name__ == "__main__":
    main()