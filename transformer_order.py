from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import logging
import asyncio
import configparser

# ENABLE LOGGING - options, DEBUG,INFO, WARNING?

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca Trading Client
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config.get('alpaca', 'api_key')
api_secret = config.get('alpaca', 'api_secret')
base_url = 'https://paper-api.alpaca.markets'

trading_client = TradingClient(api_key, api_secret, paper=True)

# Trading variables

trading_pair = 'BTC/USD'
qty_to_trade = 1

# Initialization
# Wait time between each bar request and model training
waitTime = 3600
data = 0
current_position, current_price = 0, 0
predicted_price = 0

async def main():
   '''
   Function to get latest asset data and check possible trade conditions
   '''
# closes all position AND also cancels all open orders
# trading_client.close_all_positions(cancel_orders=True)
# logger.info("Closed all positions")

   while True:
       logger.info('--------------------------------------------')
       # load model
       pred = stockPred()
       
       # get prediced price
       global predicted_price
       predicted_price = pred.predictModel()
       logger.info("Predicted Price is {0}".format(predicted_price))
       l1 = loop.create_task(check_condition())
       await asyncio.wait([l1])
       # operate trading every 1 hour
       await asyncio.sleep(waitTime)


async def check_condition():
   '''
   Strategy:

- If the predicted price an hour from now is above the current price and we do not have a position, buy
- If the predicted price an hour from now is below the current price and we do have a position, sell
   '''
   global current_position, current_price, predicted_price
   current_position = get_positions()
   logger.info("Current Price is: {0}".format(current_price))
   logger.info("Current Position is: {0}".format(current_position))

# If we do not have a position and current price is less than the predicted price place a market buy order

   if float(current_position) <= 0.01 and current_price < predicted_price:
       logger.info("Placing Buy Order")
       buy_order = await post_alpaca_order('buy')
       if buy_order:  
           logger.info("Buy Order Placed")

# If we do have a position and current price is greater than the predicted price place a market sell order

   if float(current_position) >= 0.01 and current_price > predicted_price:
       logger.info("Placing Sell Order")
       sell_order = await post_alpaca_order('sell')
       if sell_order:
           logger.info("Sell Order Placed")
           
           
async def post_alpaca_order(side):
   '''
   Post an order to Alpaca
   '''
   try:
       if side == 'buy':
           market_order_data = MarketOrderRequest(
               symbol="BTC/USD",
               qty=qty_to_trade,
               side=OrderSide.BUY,
               time_in_force=TimeInForce.GTC
           )
           buy_order = trading_client.submit_order(
               order_data=market_order_data
           )
           return buy_order
       else:
           market_order_data = MarketOrderRequest(
               symbol="BTC/USD",
               qty=current_position,
               side=OrderSide.SELL,
               time_in_force=TimeInForce.GTC
           )
           sell_order = trading_client.submit_order(
               order_data=market_order_data
           )
           return sell_order
   except Exception as e:
       logger.exception(
           "There was an issue posting order to Alpaca: {0}".format(e))
       return False
def get_positions():
   positions = trading_client.get_all_positions()
   global current_position
   for p in positions:
       if p.symbol == "BTC/USD":
           current_position = p.qty
           return current_position
   return current_position

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()



