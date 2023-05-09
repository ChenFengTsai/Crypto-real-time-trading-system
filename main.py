# import argparse
# from market_minute import DataCollector

# #symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD']

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--symbols', nargs='+', help='Trading symbols (e.g., BTC/USD ETH/USD)')
#     args = parser.parse_args()

#     if not args.symbols:
#         print('Please provide trading symbols.')
#         return

#     data_collector = DataCollector('config.ini', args.symbols)
#     data_collector.run()

# if __name__ == "__main__":
#     main()
# import argparse
# import asyncio
# from streaming_trade import AlpacaStream
# from market_minute import DataCollector

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--symbols', nargs='+', help='Trading symbols (e.g., BTC/USD ETH/USD)')
#     args = parser.parse_args()

#     if not args.symbols:
#         print('Please provide trading symbols.')
#         return

#     # alpaca_stream = AlpacaStream('config.ini', args.symbols)
#     # data_collector = DataCollector('config.ini', args.symbols)

#     # tasks = [alpaca_stream.start_stream(), data_collector.run()]
#     # #tasks = [data_collector.run()]

#     # asyncio.run(asyncio.gather(*tasks))
#     async def run_tasks():
#         alpaca_stream = AlpacaStream('config.ini', args.symbols)
#         data_collector = DataCollector('config.ini', args.symbols)

#         #tasks = [alpaca_stream.start_stream(), data_collector.run()]
#         tasks = [data_collector.run()]
#         await asyncio.gather(*tasks)

#     loop = asyncio.get_event_loop()
#     try:
#         loop.run_until_complete(run_tasks(args))
#     finally:
#         loop.close()

# if __name__ == "__main__":
#     main()
import argparse
import asyncio
from streaming_trade import AlpacaStream
from market_minute import DataCollector

async def run_tasks(args):
    alpaca_stream = AlpacaStream('config.ini', args.symbols)
    data_collector = DataCollector('config.ini', args.symbols)

    await asyncio.gather(alpaca_stream.start_stream(), data_collector.run())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', help='Trading symbols (e.g., BTC/USD ETH/USD)')
    args = parser.parse_args()

    if not args.symbols:
        print('Please provide trading symbols.')
        return

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_tasks(args))
    finally:
        loop.close()

if __name__ == "__main__":
    main()


