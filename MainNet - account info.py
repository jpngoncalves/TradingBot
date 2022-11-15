import config
from binance.client import Client

client = Client(config.API_KEY, config.API_SECRET)

TRADE_PAR = 'ADA'

asset = client.get_asset_balance(asset=(TRADE_PAR))
print('Total Balance |', TRADE_PAR, asset['free'])
print('Total Balance |', TRADE_PAR, asset['free'], file=open('Logs/log-accountinfo.txt', 'a'))

balance = client.get_account()
#print(balance)

tickers = client.get_all_tickers()
#print(tickers)

exchange_info = client.get_exchange_info()
#print(exchange_info)

permissions = client.get_account_api_permissions()
print(permissions)
print(permissions, file=open('Logs/log-accountinfo.txt', 'a'))
