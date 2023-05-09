import argparse
from streaming_trade import AlpacaStream
from market_minute import DataCollector
from live_plot import AlpacaStream_Plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', help='Trading symbols (e.g., BTC/USD ETH/USD)')
    parser.add_argument('--func', choices=['stream', 'collector', 'plot'], help='Function to run')
    args = parser.parse_args()

    if not args.symbols:
        print('Please provide trading symbols.')
        return

    if args.func == 'stream':
        alpaca_stream = AlpacaStream('config.ini', args.symbols)
        alpaca_stream.start_stream()
    elif args.func == 'collector':
        data_collector = DataCollector('config.ini', args.symbols)
        data_collector.run()
    
    elif args.func == 'plot':
        alpaca_stream_plot = AlpacaStream_Plot('config.ini', args.symbols)
        alpaca_stream_plot.start_stream()
    else:
        print('Invalid class argument.')

if __name__ == "__main__":
    main()


