import json
import requests
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from strategies import ChronosStrategy, BuyAndHoldStrategy, SellAndHoldStrategy
from backtesting import backtest
from evaluation import evaluate_strategy

# Function to load variables from a JSON file
def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def mean_imputation_without_leakage(srs, high, low):
    filled_srs = srs.copy()
    for index in range(len(srs)):
        if pd.isnull(srs.iloc[index]):
            valid_values = srs.where((srs >= low[index]) & (srs <= high[index])).dropna()
            if len(valid_values) > 0:
                filled_srs.iloc[index] = valid_values.expanding().mean().iloc[-1]
            else:
                filled_srs.iloc[index] = np.nan  # In case there are no valid values to compute the mean
    return filled_srs

def clean_and_fill_data(data):
    # Check and replace bad values with NaN
    bad_open_condition = (data['open'] < data['low']) | (data['open'] > data['high'])
    bad_close_condition = (data['close'] < data['low']) | (data['close'] > data['high'])
    
    data.loc[bad_open_condition, 'open'] = np.nan
    data.loc[bad_close_condition, 'close'] = np.nan

    # Check for NaN values in 'volume' and replace with NaN (if not already NaN)
    data['volume'] = data['volume'].replace(0, np.nan)

    # Fill NaN values using the mean_imputation_without_leakage function
    data['open'] = mean_imputation_without_leakage(data['open'], data['high'], data['low'])
    data['close'] = mean_imputation_without_leakage(data['close'], data['high'], data['low'])
    data['volume'] = data['volume'].fillna(method='ffill').fillna(method='bfill')

    return data

# Make API call function
def make_api_call(base_url, endpoint="", method="GET", **kwargs):
    # Construct the full URL
    full_url = f'{base_url}{endpoint}'

    # Make the API call
    response = requests.request(method=method, url=full_url, **kwargs)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        return response
    else:
        # If the request was not successful, raise an exception with the error message
        raise Exception(f'API request failed with status code {response.status_code}: {response.text}')

# Get Binance historical data function
def get_binance_historical_data(symbol, interval, start_date, end_date):
    # Define basic parameters for call
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/klines'
    method = 'GET'
    
    # Set the start time parameter in the params dictionary
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1500,
        'startTime': start_date,
        'endTime': end_date
    }

    # Make initial API call to get candles
    response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    candles_data = []

    while len(response.json()) > 0:
        # Append the received candles to the list
        candles_data.extend(response.json())

        # Update the start time for the next API call
        params['startTime'] = candles_data[-1][0] + 1 # last candle open_time + 1ms
        
        if params['endTime'] != "" and params['startTime'] > params['endTime']:
            break

        # Make the next API call
        response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    # Wrap the candles data as a pandas DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    dtype = {
        'open_time': 'datetime64[ms, Asia/Jerusalem]',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'close_time': 'datetime64[ms, Asia/Jerusalem]',
        'quote_asset_volume': 'float64',
        'number_of_trades': 'int64',
        'taker_buy_base_asset_volume': 'float64',
        'taker_buy_quote_asset_volume': 'float64',
        'ignore': 'float64'
    }
    
    df = pd.DataFrame(candles_data, columns=columns)
    df = df.astype(dtype)

    return df

if __name__ == "__main__":
    # Load configuration from JSON file
    config = load_config('config.json')

    backtest_values = config['backtest']
    
    symbol = backtest_values['symbol']
    context_symbol = backtest_values['context_symbol']
    interval = backtest_values['interval']
    start_date = backtest_values['start_date']
    end_date = backtest_values['end_date']
    window = backtest_values['window']
    balance = backtest_values['balance']
    clean_the_data = backtest_values['clean_the_data']
    save_to_csv = backtest_values['save_to_csv']
    slippage_factor = backtest_values['slippage_factor']
    comission = backtest_values['comission']
    sl_rate = backtest_values['sl_rate']

    # Convert start and end dates to timestamps
    start_date = int(datetime(**start_date).timestamp() * 1000)
    end_date = ''
    if end_date['year'] != -1:
        end_date = int(datetime(**start_date).timestamp() * 1000)

    # Get data
    data = get_binance_historical_data(symbol, interval, start_date, end_date)
    
    if clean_the_data:
        data = clean_and_fill_data(data)
        
    context_df = get_binance_historical_data(context_symbol, interval, start_date, end_date)
    
    if clean_the_data:
        context_df = clean_and_fill_data(context_df)
    
    context = [data.high.values, data.low.values, context_df.close.values, data.volume.values, context_df.volume.values]
    strategy = ChronosStrategy(window)
    data["strategy_signal"] = strategy.calc_signal(data.copy(deep=True),context)

    BAH_strategy = BuyAndHoldStrategy(WINDOW=window)
    SAH_strategy = SellAndHoldStrategy(WINDOW=window)
    data["BAH_signal"] = BAH_strategy.calc_signal(data.copy(deep=True))
    data["SAH_signal"] = SAH_strategy.calc_signal(data.copy(deep=True))
    
    backtest_df = backtest(data=data.copy(deep=True), strategy=strategy, signals=data["strategy_signal"],
                    starting_balance=balance, slippage_factor=slippage_factor, comission=comission, sl_rate=sl_rate)
    print("\nPortfolio value:", backtest_df["portfolio_value"].iloc[-1], "\n")
    evaluate_strategy(backtest_df, "Chronos Strategy")
    BAH_backtest_df = backtest(data=data.copy(deep=True), strategy=BAH_strategy, signals=data["BAH_signal"],
                    starting_balance=balance, slippage_factor=slippage_factor, comission=comission, sl_rate=None)
    print("\nPortfolio value:", BAH_backtest_df["portfolio_value"].iloc[-1], "\n")
    evaluate_strategy(BAH_backtest_df, "Buy and Hold Strategy")
    SAH_backtest_df = backtest(data=data.copy(deep=True), strategy=SAH_strategy, signals=data["SAH_signal"],
                    starting_balance=balance, slippage_factor=slippage_factor, comission=comission, sl_rate=None)
    print("\nPortfolio value:", SAH_backtest_df["portfolio_value"].iloc[-1], "\n")
    evaluate_strategy(SAH_backtest_df, "Sell and Hold Strategy")
    
    if save_to_csv:
        backtest_df["BAH_portfolio_value"] = BAH_backtest_df["portfolio_value"]
        backtest_df["SAH_portfolio_value"] = SAH_backtest_df["portfolio_value"]
        backtest_df.to_csv('backtest_df.csv')
    
    # Creating the plot for percent change in portfolio value
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_df['open_time'], backtest_df['portfolio_value'], label='Portfolio Value', linewidth=1, alpha=0.8, color='green')
    plt.plot(BAH_backtest_df['open_time'], BAH_backtest_df['portfolio_value'], label = 'Buy and Hold', linewidth=1, alpha=0.5, color='blue')
    plt.plot(SAH_backtest_df['open_time'], SAH_backtest_df['portfolio_value'], label = 'Sell and Hold', linewidth=1, alpha=0.5, color='red')

    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value in USD')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()
