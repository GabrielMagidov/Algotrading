from mimetypes import init
import pandas as pd
import numpy as np
import random
from models import ActionType, Position, PositionType, StrategySignal
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Any, List
import json
import platform # To check the operating system

# Load the currect chronos based on the operating system
if platform.system() == 'Darwin': # mac 
    from chronos_mlx import ChronosPipeline
else: # windows
    from chronos import ChronosPipeline



# Function to load variables from a JSON file
def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

class BaseStrategy(ABC):
    def __init__(self) -> None:
        self.sl = None
        self.tp = None
        super().__init__()

    def set_sl_tp(self, sl, tp):
        self.sl = sl
        self.tp = tp

    @abstractmethod
    def calc_signal(self, data: pd.DataFrame):
        pass

    def calc_qty(self, real_price: float, balance: float, action: ActionType, **kwargs) -> float:
        if action == ActionType.BUY:
            qty = balance / real_price

        elif action == ActionType.SELL:
            qty = balance / real_price

        return qty

    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        tp_res = self.is_take_profit(row, position)
        sl_res = self.is_stop_loss(row, position)
        if tp_res == None:
            return sl_res
        elif sl_res == None:
            return tp_res
        else:
            p = random.random()
            if p > 0.5:
                return sl_res
            else:
                return tp_res

    def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the stop-loss level.

        Returns:
            Tuple[float, float, ActionType] or None: If stop-loss is triggered, returns a tuple containing quantity and stop-loss price and action type, otherwise returns None.
        """
        if position.type == PositionType.LONG and self.sl != None:
            if row['low'] <= self.sl:
                return position.qty, self.sl, ActionType.SELL
        if position.type == PositionType.SHORT and self.sl != None:
            if row['high'] >= self.sl:
                return position.qty, self.sl, ActionType.BUY

    def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the take-profit level.

        Returns:
            Tuple[float, float, ActionType] or None: If take-profit is triggered, returns a tuple containing quantity and take-profit price and action type, otherwise returns None.
        """
        if position.type == PositionType.LONG and self.tp != None:
            if row['low'] >= self.tp:
                return position.qty, self.tp, ActionType.SELL
        if position.type == PositionType.SHORT and self.tp != None:
            if row['high'] <= self.tp:
                return position.qty, self.tp, ActionType.BUY

class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self, WINDOW):
        super().__init__()
        self.WINDOW = WINDOW
    
    def calc_signal(self, data: pd.DataFrame) -> pd.Series:
        data['strategy_signal'] = StrategySignal.DO_NOTHING
        data.iloc[self.WINDOW, data.columns.get_loc('strategy_signal')] = ActionType.BUY
        data.iloc[-1, data.columns.get_loc('strategy_signal')] = ActionType.SELL
        return data['strategy_signal'] 

class SellAndHoldStrategy(BaseStrategy):
    def __init__(self, WINDOW) -> pd.Series:
        super().__init__()
        self.WINDOW = WINDOW
    
    def calc_signal(self, data: pd.DataFrame) -> pd.Series:
        data['strategy_signal'] = StrategySignal.DO_NOTHING
        data.iloc[self.WINDOW, data.columns.get_loc('strategy_signal')] = ActionType.SELL
        data.iloc[-1, data.columns.get_loc('strategy_signal')] = ActionType.BUY
        return data['strategy_signal']
class ChronosStrategy(BaseStrategy):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
    
    def MacdDiff(self, close, fast_length, slow_length):
        fast_ema = close.ewm(span=fast_length, min_periods=fast_length, adjust=False).mean()
        slow_ema = close.ewm(span=slow_length, min_periods=slow_length, adjust=False).mean()
        
        return fast_ema - slow_ema

    def SmoothSrs(self, srs, smoothing_f):
        smoothed_srs = pd.Series([0] * len(srs), index=srs.index)
        smoothed_srs.iloc[0] = srs.iloc[0]

        for i in range(1, len(srs)):
            if pd.isna(smoothed_srs.iloc[i-1]):
                smoothed_srs.iloc[i] = srs.iloc[i]
            else:
                smoothed_srs.iloc[i] = smoothed_srs.iloc[i-1] + \
                    smoothing_f * (srs.iloc[i] - smoothed_srs.iloc[i-1])

        return smoothed_srs

    def NormalizeSmoothSrs(self, series, window_length, smoothing_f):
        lowest = series.rolling(window_length).min()
        highest_range = series.rolling(window_length).max() - lowest
        normalized_series = series.copy()
        if (highest_range > 0).any():
            normalized_series = (series - lowest) / highest_range * 100
        else:
            normalized_series = pd.Series([pd.NA] * len(series), index=series.index)
        normalized_series = normalized_series.ffill()
        smoothed_series = self.SmoothSrs(normalized_series, smoothing_f)
        return smoothed_series

    def calc_STC(self, data: pd.DataFrame, stc_length, fast_length, slow_length, AAA) -> pd.Series:
        macd_diff = self.MacdDiff(data["close"], fast_length, slow_length)
        normalized_macd = self.NormalizeSmoothSrs(macd_diff, stc_length, AAA)
        final_stc = self.NormalizeSmoothSrs(normalized_macd, stc_length, AAA)
        return final_stc

    def predict_model(self, data: pd.DataFrame, context: np.ndarray[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        config = load_config('config.json')
        
        # Load chronos config
        chronos_values = config['chronos']
        
        is_mac = platform.system() == 'Darwin'
        if platform.system() == 'Linux':
            raise NotImplementedError("Our Code does isn't supported on Linux yet.")
        prediction_length = chronos_values['prediction_length']
        num_samples = chronos_values['num_samples']
        temperature = chronos_values['temperature']
        top_k = chronos_values['top_k']
        top_p = chronos_values['top_p']
        
        if is_mac:
            pipeline = ChronosPipeline.from_pretrained(
                model_name_or_path="amazon/chronos-t5-small",
                dtype="bfloat16",
            )
            
        else:
            pipeline = ChronosPipeline.from_pretrained(
                model_name_or_path="amazon/chronos-t5-small",
                device_map = "cuda", # use "cpu" for CPU inference
                dtype = torch.bfloat16, # use torch.float32 for FP32 inference
            )
            context = torch.Tensor(context) # Convert context to torch tensor for GPU inference
            
        forecast = pipeline.predict(
            context = context,
            prediction_length = prediction_length,
            num_samples = num_samples,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
        )

        forecast_index = range(len(data), len(data) + 2)
        low, median, high = np.quantile(forecast[0], [0.1, 0.5, 0.9], axis=0)
        return low, median, high

    def calc_signal(self, data: pd.DataFrame, context) -> pd.Series:
        config = load_config('config.json')
        
        stc_values = config['stc']
        
        stc_length = stc_values['stc_length']
        fast_length = stc_values['fast_length']
        slow_length = stc_values['slow_length']
        aaa = stc_values['aaa']
        over_sold = stc_values['over_sold']
        over_bought = stc_values['over_bought']       
        
        data["STC"] = self.calc_STC(data, stc_length, fast_length, slow_length, aaa)
        data["strategy_signal"] = StrategySignal.DO_NOTHING
        buy_num = 0
        sell_num = 0
        for i in range(self.window, len(data)):
            if i % 50 == 0:
                print(f'At row {i}')
            low, median, high = self.predict_model(data.iloc[i - self.window:i + 1], context)
            
            if (median[0] > data.iloc[i]['close']) and (data.iloc[i, data.columns.get_loc('STC')] < over_bought):
                data.iloc[i, data.columns.get_loc('strategy_signal')] = ActionType.BUY
                buy_num += 1
            elif median[0] < data.iloc[i]['close'] and (data.iloc[i, data.columns.get_loc('STC')] > over_sold):
                data.iloc[i, data.columns.get_loc('strategy_signal')] = ActionType.SELL
                sell_num += 1
        print(f'buy: {buy_num}, sell: {sell_num}')
        data["strategy_signal"] = data["strategy_signal"].shift()
        return data['strategy_signal']