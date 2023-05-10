import pandas as pd
import time
import alpaca_trade_api as api
import configparser
import random


# Define a function to calculate EMA values
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Define a function to generate trading signals
def generate_signals(data, short_window, long_window):

    # Calculate EMA values for short and long windows
    ema_short = calculate_ema(data, short_window)
    ema_long = calculate_ema(data, long_window)

    # # Generate trading signals based on EMA crossovers
    # signals.loc[short_window:, 'signal'] = \
    #     (ema_short[short_window:] > ema_long[short_window:]).astype(int)

    # # Calculate the difference between consecutive signals to find the signal changes
    # signals['positions'] = signals['signal'].diff()
    
    

    return signals


short_window = 20
long_window = 50

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config.get('alpaca', 'API_KEY')
api_secret = config.get('alpaca', 'SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

alpaca = api.REST(api_key, api_secret, BASE_URL)
while True:
    data = pd.read_csv('btcusd_bar_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    generate_signals(data['close'], short_window, long_window)
    time.sleep(1)