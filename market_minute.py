import asyncio
import alpaca_trade_api as tradeapi
import configparser
import csv
from alpaca.data.live import CryptoDataStream
import argparse

class DataCollector:
    def __init__(self, config_path, symbols):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.api_key = self.config.get('alpaca', 'API_KEY')
        self.api_secret = self.config.get('alpaca', 'SECRET_KEY')
        self.symbols = symbols
        self.crypto_stream = None

    async def subscribe_data(self):
        tasks = []
        for symbol in self.symbols:
            tasks.append(self.crypto_stream.subscribe_bars(self.bar_callback, symbol))
        await asyncio.gather(*tasks)

    async def bar_callback(self, bar):
        row = [value for _, value in bar]
        symbol = row[0].lower().replace('/', '')
        filename = f"{symbol}_bar_data.csv"

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if csvfile.tell() == 0:
                header = [property_name for property_name, _ in bar]
                writer.writerow(header)

            row = [value for _, value in bar]
            writer.writerow(row)

    def run(self):
        self.crypto_stream = CryptoDataStream(self.api_key, self.api_secret)
        for symbol in self.symbols:
            self.crypto_stream.subscribe_bars(self.bar_callback, symbol)
        self.crypto_stream.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', help='Trading symbols (e.g., BTC/USD ETH/USD)')
    args = parser.parse_args()

    if not args.symbols:
        print('Please provide trading symbols.')
        return

    data_collector = DataCollector('config.ini', args.symbols)
    data_collector.run()

if __name__ == "__main__":
    main()


    #symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD']
    
