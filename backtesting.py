import pandas as pd
import yfinance
import requests
import numpy as np
from datetime import datetime
from models import ActionType, PositionType, Position, StrategySignal
from strategies import BaseStrategy
import matplotlib.pyplot as plt


def calc_realistic_price(row: pd.Series, action_type: ActionType, slippage_factor):
    slippage_rate = ((row['close'] - row['open']) /
                     row['open']) / slippage_factor
    slippage_price = row['open'] + row['open'] * slippage_rate

    if action_type == ActionType.BUY:
        return max(slippage_price, row['open'])
    else:
        return min(slippage_price, row['open'])

def backtest(data: pd.DataFrame, strategy: BaseStrategy, signals: pd.Series ,starting_balance: int, slippage_factor, comission, sl_rate = None) -> pd.DataFrame:

    def enter_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position_type: PositionType) -> Position:
        """
        Enters a new position in the trading strategy.

        Args:
            data (pd.DataFrame): The data frame containing the trading data.
            index (int): The index of the current row in the data frame.
            row (pd.Series): The current row of data.
            curr_qty (float): The current quantity of the asset held.
            curr_balance (float): The current balance available for trading.
            position_type (PositionType): The type of position to enter (LONG or SHORT).

        Returns:
            Position: The newly created position.

        Raises:
            None

        """
        if position_type == PositionType.LONG:
            buy_price = calc_realistic_price(
                row, ActionType.BUY, slippage_factor=slippage_factor)
            qty_to_buy = strategy.calc_qty(
                buy_price, curr_balance, ActionType.BUY)
            position = Position(qty_to_buy, buy_price, position_type)
            data.loc[index, 'qty'] = curr_qty + qty_to_buy
            data.loc[index, 'balance'] = curr_balance - \
                qty_to_buy * buy_price

        elif position_type == PositionType.SHORT:
            sell_price = calc_realistic_price(
                row, ActionType.SELL, slippage_factor=slippage_factor)
            qty_to_sell = strategy.calc_qty(
                sell_price, curr_balance, ActionType.SELL)
            position = Position(qty_to_sell, sell_price, position_type)
            data.loc[index, 'qty'] = curr_qty - qty_to_sell
            data.loc[index, 'balance'] = curr_balance + \
                qty_to_sell * sell_price

        return position

    def close_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position: Position):
        if position.type == PositionType.LONG:
            sell_price = calc_realistic_price(
                row, ActionType.SELL, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty - position.qty
            data.loc[index, 'balance'] = curr_balance + \
                position.qty * sell_price
            if sl_rate is not None:
                strategy.set_sl_tp((1 + sl_rate) * sell_price,
                               (1 + 2 * sl_rate) * sell_price)

        elif position.type == PositionType.SHORT:
            buy_price = calc_realistic_price(
                row, ActionType.BUY, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty + position.qty
            data.loc[index, 'balance'] = curr_balance - \
                position.qty * buy_price
            if sl_rate is not None:
                strategy.set_sl_tp((1 + sl_rate) * buy_price,
                               (1 + 2 * sl_rate) * buy_price)

    # initialize df
    data['qty'] = 0.0
    data['balance'] = 0.0
    counter = 0
    
    data['strategy_signal'] = signals

    # Loop through the data to calculate portfolio value
    position: Position = None
    data.reset_index(inplace=True)
    max_candle = data.shape[0]

    for index, row in data.iterrows():
        curr_qty = data.loc[index - 1, 'qty'] if index > 0 else 0
        curr_balance = data.loc[index - 1,
                                'balance'] if index > 0 else starting_balance
        if position is not None:
            if curr_qty != 0:
                sl_tp_res = strategy.check_sl_tp(
                    data.iloc[index - 1], position)
                if sl_tp_res is not None:
                    sl_tp_qty, sl_tp_price, sl_tp_action = sl_tp_res
                    if sl_tp_action == ActionType.BUY:
                        curr_balance = curr_balance - sl_tp_qty * sl_tp_price
                        curr_qty = curr_qty + sl_tp_qty
                        position = None

                    elif sl_tp_action == ActionType.SELL:
                        curr_balance = curr_balance + sl_tp_qty * sl_tp_price
                        curr_qty = curr_qty - sl_tp_qty
                        position = None

        if row['strategy_signal'] == ActionType.BUY:
            if position is not None and position.type == PositionType.SHORT:
                row['strategy_signal'] = StrategySignal.CLOSE_SHORT
            elif curr_qty == 0:
                row['strategy_signal'] = StrategySignal.ENTER_LONG
        elif row['strategy_signal'] == ActionType.SELL:
            if position is not None and position.type == PositionType.LONG:
                row['strategy_signal'] = StrategySignal.CLOSE_LONG
            elif curr_qty == 0:
                row['strategy_signal'] = StrategySignal.ENTER_SHORT

        # Close position at end of trade
        if index + 1 == max_candle and position is not None:
            close_position(data, index, row, curr_qty, curr_balance, position)
            position = None

        # Handle enter long signal
        elif row['strategy_signal'] == StrategySignal.ENTER_LONG:
            counter += 1
            position = enter_position(
                data, index, row, curr_qty, curr_balance, PositionType.LONG)

        # Handle enter short signal
        elif row['strategy_signal'] == StrategySignal.ENTER_SHORT:
            counter += 1
            position = enter_position(
                data, index, row, curr_qty, curr_balance, PositionType.SHORT)

        # Handle close long or short signal
        elif row['strategy_signal'] in [StrategySignal.CLOSE_LONG, StrategySignal.CLOSE_SHORT] and position is not None:
            close_position(data, index, row, curr_qty, curr_balance, position)
            position = None
            strategy.set_sl_tp(None, None)

        else:
            data.loc[index, 'qty'] = curr_qty
            data.loc[index, 'balance'] = curr_balance

    # Calculate portfolio value
    data['portfolio_value'] = data['close'] * data['qty'] + data['balance'] - counter * 2 * comission
    print(f"Number of Trades : {counter}") 
    return data