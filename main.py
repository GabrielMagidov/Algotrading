import json
import requests
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
from strategies import ChronosStrategy, BuyAndHoldStrategy, SellAndHoldStrategy
from backtesting import backtest
from evaluation import evaluate_strategy

# Function to load variables from a JSON file
def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

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

    SYMBOL = config['SYMBOL']
    CONTEXT_SYMBOL = config['CONTEXT_SYMBOL']
    INTERVAL = config['INTERVAL']
    START_DATE = config['START_DATE']
    END_DATE = config['END_DATE']
    WINDOW = config['WINDOW']
    BALANCE = config['BALANCE']

    # Convert start and end dates to timestamps
    start_date = int(datetime(**START_DATE).timestamp() * 1000)
    end_date = ''
    if END_DATE['year'] != -1:
        end_date = int(datetime(**END_DATE).timestamp() * 1000)

    # Get data
    data = get_binance_historical_data(SYMBOL, INTERVAL, start_date, end_date)
    context_df = get_binance_historical_data(CONTEXT_SYMBOL, INTERVAL, start_date, end_date)
    
    context = [data.high.values, data.low.values, context_df.close.values, data.volume.values, context_df.volume.values]
    strategy = ChronosStrategy(WINDOW)
    data["strategy_signal"] = strategy.calc_signal(data.copy(deep=True),context)

    BAH_strategy = BuyAndHoldStrategy(WINDOW=WINDOW)
    SAH_strategy = SellAndHoldStrategy(WINDOW=WINDOW)
    data["BAH_signal"] = BAH_strategy.calc_signal(data.copy(deep=True))
    data["SAH_signal"] = SAH_strategy.calc_signal(data.copy(deep=True))
    
    SLIPPAGE_FACTOR = config["SLIPPAGE_FACTOR"]
    COMISSION = config["COMISSION"]
    SL_RATE = config['SL_RATE']
    
    backtest_df = backtest(data=data.copy(deep=True), strategy=strategy, signals=data["strategy_signal"],
                    starting_balance=BALANCE, slippage_factor=SLIPPAGE_FACTOR, comission=COMISSION, sl_rate=SL_RATE)
    print("\nPortfolio value:", backtest_df["portfolio_value"].iloc[-1], "\n")
    evaluate_strategy(backtest_df, "Chronos Strategy")
    BAH_backtest_df = backtest(data=data.copy(deep=True), strategy=BAH_strategy, signals=data["BAH_signal"],
                    starting_balance=BALANCE, slippage_factor=SLIPPAGE_FACTOR, comission=COMISSION, sl_rate=None)
    print("\nPortfolio value:", BAH_backtest_df["portfolio_value"].iloc[-1], "\n")
    evaluate_strategy(BAH_backtest_df, "Buy and Hold Strategy")
    SAH_backtest_df = backtest(data=data.copy(deep=True), strategy=SAH_strategy, signals=data["SAH_signal"],
                    starting_balance=BALANCE, slippage_factor=SLIPPAGE_FACTOR, comission=COMISSION, sl_rate=None)
    print("\nPortfolio value:", SAH_backtest_df["portfolio_value"].iloc[-1], "\n")
    evaluate_strategy(SAH_backtest_df, "Sell and Hold Strategy")
    
    if config["SAVE_TO_CSV"]:
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
