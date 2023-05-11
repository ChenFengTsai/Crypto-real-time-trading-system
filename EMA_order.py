import pandas as pd

# Define a function to calculate EMA values
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Define a function to generate trading signals
def generate_signals(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0

    # Calculate EMA values for short and long windows
    ema_short = calculate_ema(data, short_window)
    ema_long = calculate_ema(data, long_window)

    # Generate trading signals based on EMA crossovers
    signals.loc[short_window:, 'signal'] = \
        (ema_short[short_window:] > ema_long[short_window:]).astype(int)

    # Calculate the difference between consecutive signals to find the signal changes
    signals['positions'] = signals['signal'].diff()

    return signals

# Example usage
# Load historical price data into a DataFrame
# Here, 'data' should contain the 'Date' and 'Close' columns
data = pd.read_csv('historical_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define the short and long EMA windows
short_window = 20
long_window = 50

# Generate trading signals
signals = generate_signals(data['Close'], short_window, long_window)

# Print the signals DataFrame
print(signals)
