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


import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *

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
        path = './model/Transformer+TimeEmbedding.hdf5'
        model = tf.keras.models.load_model(path,
                                        custom_objects={'Time2Vector': Time2Vector, 
                                                        'SingleAttention': SingleAttention,
                                                        'MultiAttention': MultiAttention,
                                                        'TransformerEncoder': TransformerEncoder})    
        # get prediced price
        global predicted_price
        logger.info("Getting bar data for {0} starting from {1}".format(trading_pair, time_diff))
        # Defining Bar data request parameters
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[trading_pair],
            timeframe=TimeFrame.Hour,
            start="2020-01-01 00:00:00")
 
        # Get the bar data from Alpaca
        data_client = CryptoHistoricalDataClient()
        bars_df = data_client.get_crypto_bars(request_params).df
        
        # change to returns to feed into model
        df = pd.DataFrame()
        df['open'] = bars_df['open'].pct_change()
        df['high'] = bars_df['high'].pct_change()
        df['low'] = bars_df['low'].pct_change()
        df['close'] = bars_df['close'].pct_change()
        df['volume'] = bars_df['volume'].pct_change()
        
        # Normalization
        min_return = min(df[['open', 'high', 'low', 'close']].min(axis=0))
        max_return = max(df[['open', 'high', 'low', 'close']].max(axis=0))
        
        df['open'] = (df['open'] - min_return) / (max_return - min_return)
        df['high'] = (df['high'] - min_return) / (max_return - min_return)
        df['low'] = (df['low'] - min_return) / (max_return - min_return)
        df['close'] = (df['close'] - min_return) / (max_return - min_return)
        
        min_volume = df['volume'].min(axis=0)
        max_volume = df['volume'].max(axis=0)
        df.loc[:, 'volume'] = (df.loc[:,'volume'] - min_volume) / (max_volume - min_volume)
        
        global current_price
        current_price = bars_df.iloc[-1]['close']
        
        # df to array
        current_input = current_input.values
        
        # get sequential input with length 128
        seq_len=128
        seq_input = [current_input[-128:]]
        seq_input = np.array(seq_input)    
        
        # get predict price
        global predicted_price
        predicted_price = model.predict(seq_input)
        logger.info("Predicted Price is {0}".format(predicted_price))

        l1 = loop.create_task(check_condition())
        await asyncio.wait([l1])
        # operate trading every 1 hour
        await asyncio.sleep(waitTime)

###### model ########
class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        # inherit from Layer
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        # Initialize weights and biases with shape (batch, seq_len)
        # for linear pattern
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),), # shape is same as the sequence length
                                    initializer='uniform', # randomized between 0-1
                                    trainable=True)
        
        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)
        # for periodic pattern
        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        # Calculate linear and periodic time features
        x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
        time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
        
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
    def get_config(self): 
        # saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config

class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        # choose glorot_uniform as weight initialization to control variance
        self.query = Dense(self.d_k, 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')
        
        self.key = Dense(self.d_k, 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')
        
        self.value = Dense(self.d_v, 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        
        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out 

class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
        
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
        self.linear = Dense(input_shape[0][-1], 
                            input_shape=input_shape, 
                            kernel_initializer='glorot_uniform', 
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear  

class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer 

    def get_config(self): 
        # saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                    'd_v': self.d_v,
                    'n_heads': self.n_heads,
                    'ff_dim': self.ff_dim,
                    'attn_heads': self.attn_heads,
                    'dropout_rate': self.dropout_rate})
        return config  

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



