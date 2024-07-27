# Algo Trading with Chronos AI and STC Indicator

This project implements an algorithmic trading strategy using Amazon's "Chronos" AI model and the Schaff Trend Cycle (STC) technical indicator to generate trading signals. The algorithm is designed to predict Bitcoin prices and make informed trading decisions.

## Features

- **Chronos AI Model**: Utilizes Amazon's AI model to predict future Bitcoin prices.
- **STC Indicator**: Filters predictions using the Schaff Trend Cycle technical indicator.
- **Backtesting**: Includes a backtesting module to evaluate the performance of the trading strategy.
- **Configurable**: Provides a configuration file to customize the algorithm parameters.

## How It Works

1. **Data Collection**: The algorithm collects 1000 hourly Bitcoin candles.
2. **Price Prediction**: The Chronos model predicts the next Bitcoin price 20 times.
3. **Signal Generation**: Predictions are filtered using the STC indicator to generate buy/sell signals.
4. **Backtesting**: The generated signals are backtested to assess the algorithm's performance.

## Chronos AI Model

Chronos is a family of pretrained time series forecasting models based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. Chronos models have been trained on a large corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes.

For details on Chronos models, training data and procedures, and experimental results, please refer to the paper: "Chronos: Learning the Language of Time Series".

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GabrielMagidov/Algotrading
   cd Algotrading
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements_{YOUR_OS}.txt

## Usage
1. **Configuration**:
   Modify the config.json file to set your desired parameters.
2. **Run the Algorithm and Backtesting**:
   ```bash
   python main.py
   ```

## Backtesting
The backtesting process evaluates the performance of the algorithm. After the backtesting completes, it plots the portfolio value for the strategy, buy and hold, and sell and hold strategies to compare. For each strategy, it prints the following metrics:
- Total Return
- Annualized Return
- Annualized Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio

**Example Output**:
```bash
Results for Chronos Strategy:
Total Return: 50.00%
Annualized Return: 12.34%
Annualized Sharpe Ratio: 1.23
Sortino Ratio: 1.56
Max Drawdown: 10.00%
Calmar Ratio: 1.23

Results for Buy and Hold Strategy:
Total Return: 40.00%
Annualized Return: 10.00%
Annualized Sharpe Ratio: 1.10
Sortino Ratio: 1.30
Max Drawdown: 12.00%
Calmar Ratio: 1.00

Results for Sell and Hold Strategy:
Total Return: 30.00%
Annualized Return: 8.00%
Annualized Sharpe Ratio: 0.90
Sortino Ratio: 1.10
Max Drawdown: 15.00%
Calmar Ratio: 0.80
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
    




