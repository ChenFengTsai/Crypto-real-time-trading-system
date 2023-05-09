import asyncio
import alpaca_trade_api as tradeapi
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config.get('alpaca', 'api_key')
api_secret = config.get('alpaca', 'api_secret')
base_url = 'https://paper-api.alpaca.markets'

async def place_orders(symbols):
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    tasks = []
    for symbol in symbols:
        tasks.append(api.submit_order(
            symbol=symbol,
            qty=100,
            side='buy',
            type='market',
            time_in_force='gtc'
        ))
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    symbols = ['BTCUSD', 'ETHUSD', 'DOGEUSD']
    asyncio.run(place_orders(symbols))
