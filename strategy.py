def calculate_ema(data, time_period):
    smoothing_factor = 2 / (time_period + 1)
    ema_values = []

    # Calculate initial SMA for the chosen time period
    initial_sma = sum(data[:time_period]) / time_period
    ema_values.append(initial_sma)

    # Calculate EMA for the remaining data points
    for i in range(time_period, len(data)):
        current_price = data[i]
        previous_ema = ema_values[-1]
        ema = (current_price - previous_ema) * smoothing_factor + previous_ema
        ema_values.append(ema)

    return ema_values

# Example usage
closing_prices = [50.2, 51.1, 52.5, 50.7, 53.2, 54.1, 55.3, 53.8, 52.6, 53.9, 55.2, 56.1]
time_period = 10
calculate_ema(closing_prices, time_period)