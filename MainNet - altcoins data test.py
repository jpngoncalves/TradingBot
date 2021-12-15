import config
from binance.client import Client
import pandas as pd
import time

client = Client(config.API_KEY, config.API_SECRET)

TRADE_PAR = 'USDT'

balance = client.get_asset_balance(asset=TRADE_PAR)
print('Total Balance |', TRADE_PAR, balance['free'])

###################### | DATA TEST | ######################

#Variables:
    #priceChangePercent
    #lastQty
    #bidQty
    #askQty
    #quoteVolume
    #volume

x = pd.DataFrame(client.get_ticker()) 
#print(x)

y = x[x.symbol.str.contains(TRADE_PAR)] 
#print(y)

z = y[~((y.symbol.str.contains('UP')) | (y.symbol.str.contains('DOWN')))]
#print(z)

z1 = z[z.priceChangePercent == z.priceChangePercent.max()]
print(z1)
print('-----------priceChangePercent-----------')

z2 = z[z.lastQty == z.lastQty.max()]
print(z2)
print('-----------lastQty-----------')

z3 = z[z.bidQty == z.bidQty.max()]
print(z3)
print('-----------bidQty-----------')

z4 = z[z.askQty == z.askQty.max()]
print(z4)
print('-----------askQty-----------')

z5 = z[z.volume == z.volume.max()]
print(z5)
print('-----------volume-----------')

z6 = z[z.quoteVolume == z.quoteVolume.max()]
print(z6)
print('-----------quoteVolume-----------')


###########################################################

def get_top_symbol():
    all_pairs = pd.DataFrame(client.get_ticker())
    relev = all_pairs[all_pairs.symbol.str.contains(TRADE_PAR)]
    non_lev = relev[~((relev.symbol.str.contains('UP')) | (relev.symbol.str.contains('DOWN')))]
    top_symbol = non_lev[non_lev.priceChangePercent == non_lev.priceChangePercent.max()] #Change here variables...
    top_symbol = top_symbol.symbol.values[0]
    return top_symbol
print(get_top_symbol())
