import pandas as pd
import numpy as np
import json

# Function to load variables from a JSON file
def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def calc_total_return(portfolio_values):
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0


def calc_annualized_return(portfolio_values):
    days_in_year = 365
    daily_trading_1h = 24
    yearly_trading_1h = days_in_year * daily_trading_1h
    portfolio_trading_days = portfolio_values.shape[0]
    portfolio_trading_years = portfolio_trading_days / yearly_trading_1h
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0])**(1/portfolio_trading_years) - 1.0


def calc_annualized_sharpe(portfolio_values: pd.Series, rf: float = 0.0):
    days_in_year = 365
    daily_trading_1h = 24
    yearly_trading_1h = days_in_year * daily_trading_1h
    annualized_return = calc_annualized_return(portfolio_values)
    annualized_std = portfolio_values.pct_change().std() * np.sqrt(yearly_trading_1h)
    if annualized_std is None or annualized_std == 0:
        return 0
    sharpe = (annualized_return - rf) / annualized_std
    return sharpe


def calc_downside_deviation(portfolio_values):
    porfolio_returns = portfolio_values.pct_change().dropna()
    return porfolio_returns[porfolio_returns < 0].std()


def calc_sortino(portfolio_values, rf=0.0):
    days_in_year = 365
    daily_trading_1h = 24
    yearly_trading_1h = days_in_year * daily_trading_1h
    down_deviation = calc_downside_deviation(
        portfolio_values) * np.sqrt(yearly_trading_1h)
    annualized_return = calc_annualized_return(portfolio_values)
    if down_deviation is None or down_deviation == 0:
        return 0
    sortino = (annualized_return - rf) / down_deviation
    return sortino


def calc_max_drawdown(portfolio_values):
    cumulative_max = portfolio_values.cummax()
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    return drawdown.max()


def calc_calmar(portfolio_values):
    max_drawdown = calc_max_drawdown(portfolio_values)
    annualized_return = calc_annualized_return(portfolio_values)
    return annualized_return / max_drawdown

def evaluate_strategy(b_df, strat_name):
    # Load configuration from JSON file
    config = load_config('config.json')

    window = config['backtest']['window']
    
    total_return = calc_total_return(
        b_df['portfolio_value'][window:])
    annualized_return = calc_annualized_return(
        b_df['portfolio_value'][window:])
    annualized_sharpe = calc_annualized_sharpe(
        b_df['portfolio_value'][window:], rf=0.04)
    sortino_ratio = calc_sortino(
        b_df['portfolio_value'][window:], rf=0.04)
    max_drawdown = calc_max_drawdown(
        b_df['portfolio_value'][window:])
    calmar_ratio = calc_calmar(b_df['portfolio_value'][window:])

    print(f"\nResults for {strat_name}:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")