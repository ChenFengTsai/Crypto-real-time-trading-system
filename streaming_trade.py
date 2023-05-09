import configparser
from alpaca_trade_api.stream import Stream
import logging

class AlpacaStream:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.api_key = self.config.get('alpaca', 'API_KEY')
        self.api_secret = self.config.get('alpaca', 'SECRET_KEY')
        self.stream = None

    async def log_quote(self, q):
        logging.info('quote %s', q)

    async def log_trade(self, t):
        logging.info('trade %s', t)

    def start_stream(self):
        self.stream = Stream(self.api_key, self.api_secret, raw_data=True)
        self.stream.subscribe_crypto_quotes(self.log_quote, 'BTC/USD')
        self.stream.subscribe_crypto_trades(self.log_trade, 'BTC/USD')

        # @self.stream.on_bar('BTC/USD')
        # async def _(bar):
        #     print('bar', bar)

        self.stream.run()

    def stop_stream(self):
        if self.stream:
            self.stream.close()

def main():
    logging.basicConfig(filename="trade_quote_log", level = logging.INFO)
    alpaca_stream = AlpacaStream('config.ini')
    alpaca_stream.start_stream()

if __name__ == '__main__':
    main()