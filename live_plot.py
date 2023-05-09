# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from mk import AlpacaStream
# from alpaca_trade_api.stream import Stream

# class CustomAlpacaStream(AlpacaStream):
#     def __init__(self, config_path):
#         super().__init__(config_path)
#         self.trades = []

#     async def append_trade(self, t):
#         self.trades.append(t['p'])
#         #print('trade', t)
        
#     def start_stream(self):
#         self.stream = Stream(self.api_key, self.api_secret, raw_data=True)
#         self.stream.subscribe_crypto_trades(self.append_trade, 'BTC/USD')

#         @self.stream.on_bar('BTC/USD')
#         async def _(bar):
#             print('bar', bar)

#         self.stream.run()

# # Set up the figure and axis
# fig, ax = plt.subplots()

# # Set up the initial plot
# x = []
# y = []
# line, = ax.plot(x, y) 
# ax.set_xlim(0, 10)
# ax.set_title("Streaming Data Plot")

# # Continuously update the plot
# while True:
#     # Generate new data or get data from the streaming source
#     new_x = time.time()  # Example: Use time as x-axis
#     alpaca_stream = CustomAlpacaStream('config.ini')
#     alpaca_stream.start_stream()
    
#     new_y = alpaca_stream.trades  # Example: Use random numbers as y-axis
#     print(new_y)
#     # Append new data to existing data
#     x.append(new_x)
#     y.append(new_y)
    
#     # Update the plot
#     line.set_data(x, y)
    
#     # # Adjust the plot limits if needed
#     ax.relim()
#     ax.set_xlim(new_x - 10, new_x) 
#     ax.autoscale_view()
    
    
#     # Pause to control the plot update speed
#     plt.pause(1)

import configparser
import matplotlib.pyplot as plt
from alpaca_trade_api.stream import Stream
from datetime import datetime
plt.style.use('seaborn')

class AlpacaStream:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.api_key = self.config.get('alpaca', 'API_KEY')
        self.api_secret = self.config.get('alpaca', 'SECRET_KEY')
        self.stream = None
        self.prices = []
        self.timestamps = []

        # Plotting
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()

    async def print_trade(self, t):
        self.prices.append(t['p'])
        self.timestamps.append(datetime.fromtimestamp(t['t'].seconds))

        self.ax.clear()
        self.ax.plot(self.timestamps, self.prices)
        self.ax.set_xlabel('Timestamp')
        self.ax.set_ylabel('Price')
        self.ax.set_title('Live Bitcoin Price Trend')
        plt.xticks(rotation=45)
        plt.pause(0.001)  # Update the plot

    def start_stream(self):
        self.stream = Stream(self.api_key, self.api_secret, raw_data=True)
        self.stream.subscribe_crypto_trades(self.print_trade, 'BTC/USD')

        @self.stream.on_bar('BTC/USD')
        async def _(bar):
            print('bar', bar)

        self.stream.run()

    def stop_stream(self):
        if self.stream:
            self.stream.close()

def main():
    alpaca_stream = AlpacaStream('config.ini')
    alpaca_stream.start_stream()

if __name__ == '__main__':
    main()
