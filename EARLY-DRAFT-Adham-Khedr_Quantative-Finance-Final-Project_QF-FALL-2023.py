import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class Tools:
    def __init__(self, data, tickers):
        self.data = data
        self.tickers = tickers

    def get_momentum_factors(self):
        data = self.data.copy()
        data = data.pct_change()
        data = data.rolling(20).mean()
        return data

    def normalizing(self):
        data = self.data.copy()
        data = data.pct_change()
        data = data.rolling(20).mean()
        zscore = (data - data.mean()) / data.std()
        return zscore

    def baskets(self):
        data = self.data.copy()
        momentum_factors = self.get_momentum_factors()
        zscore = self.normalizing()
        baskets = []

        for i in range(0, momentum_factors.shape[1]):
            basket = momentum_factors.iloc[:, i]
            basket = basket[basket > 0]
            baskets.append(basket)

        for i in range(0, len(baskets)):
            baskets[i] = baskets[i].sort_values(ascending=False)

        basket_dict = {}
        for i in range(0, len(baskets)):
            basket_dict[self.tickers[i]] = baskets[i]

        return basket_dict


def load_csv_file(IVV_holdings):
    ivv_holdings_csv = pd.read_csv(IVV_holdings)
    ivv_tickers = ivv_holdings_csv["Ticker"].dropna()
    ivv_tickers = [t for t in ivv_tickers if isinstance(t, str)]
    ticker_remove = ['BFB', 'UBFUT', 'ESZ3', 'XTSLA', 'BRKB', '\xa0']
    ticker = [t for t in ivv_tickers if t not in ticker_remove]
    return ticker


def download_data(ticker):
    ivv_data = yf.download(ticker, period='10y')
    ivv_data = ivv_data.dropna(how='all', axis=1)

    # Set the index to a DatetimeIndex if it is not already
    if not isinstance(ivv_data.index, pd.DatetimeIndex):
        ivv_data.index = pd.to_datetime(ivv_data.index)

    return ivv_data


def calculate_historical_returns(ivv_data, tickers):
    adj_close_data = ivv_data['Adj Close']  # Extract Adjusted Close data

    # Initialize dictionaries
    hist_rets_dict, long_ret_dict, shrt_ret_dict = {}, {}, {}

    for ticker in tickers:
        if ticker not in adj_close_data.columns:
            print(f"{ticker}: No data found, symbol may be delisted")
            continue

        ticker_data = adj_close_data[ticker]
        ticker_hist_rets = ticker_data.pct_change().resample("M").sum().shift(-1)
        ticker_long_ret = ticker_hist_rets.sum()
        ticker_shrt_ret = ticker_hist_rets.mean()

        hist_rets_dict[ticker] = ticker_hist_rets
        long_ret_dict[ticker] = ticker_long_ret
        shrt_ret_dict[ticker] = ticker_shrt_ret

    # Convert dictionaries to DataFrames
    hist_rets = pd.DataFrame(hist_rets_dict)
    long_ret = pd.DataFrame(long_ret_dict, index=[0])
    shrt_ret = pd.DataFrame(shrt_ret_dict, index=[0])

    return hist_rets, long_ret, shrt_ret


# Load the CSV file
ticker = load_csv_file("IVV_holdings.csv")

# Download data using yfinance
ivv_data = download_data(ticker)

# Historical Returns Calculation
hist_rets, long_ret, shrt_ret = calculate_historical_returns(ivv_data, ticker)

# Calculate Portfolio Average Monthly Return
portfolio_monthly_avg_return = hist_rets.mean(axis=1)

# Placeholder for ETF Benchmark Return - Replace with actual ETF return data
etf_benchmark_return = pd.Series([0.02] * len(portfolio_monthly_avg_return), index=portfolio_monthly_avg_return.index)  # Example values

# Calculate Cumulative Returns
portfolio_cumulative_return = (1 + portfolio_monthly_avg_return).cumprod() - 1
etf_cumulative_return = (1 + etf_benchmark_return).cumprod() - 1

# Combine into 'tot_ret' DataFrame
tot_ret = pd.DataFrame({
    'Portfolio': portfolio_monthly_avg_return,
    'Bench': etf_benchmark_return,
    'Return[Cumulative]': portfolio_cumulative_return,
    'ETF[Cumulative]': etf_cumulative_return
})

# Plotting Cumulative Returns
plt.figure(figsize=(20, 5))
plt.plot(tot_ret.index, tot_ret['Return[Cumulative]'], label='Portfolio Cumulative Returns', color='green')
plt.plot(tot_ret.index, tot_ret['ETF[Cumulative]'], label='ETF Cumulative Returns', color='red')
plt.title('Cumulative Portfolio Return vs ETF')
plt.legend()
plt.show()

# 2. Monthly Return: Long Picks vs Short Picks vs ETF
plt.figure(figsize=(18, 4))
plt.plot(long_ret.index, long_ret.mean(axis=1), label='Long Picks', color='blue')
plt.plot(shrt_ret.index, shrt_ret.mean(axis=1), label='Short Picks', color='yellow')
plt.plot(tot_ret.index, tot_ret['Bench'], label='ETF', color='red')
plt.title('Monthly Return: Long Picks vs Short Picks vs ETF')
plt.legend()
plt.show()

# 3. Cumulative Portfolio Return vs ETF
plt.figure(figsize=(20, 5))
plt.plot(tot_ret.index, tot_ret['Return[Cumulative]'], label='Portfolio Cumulative Returns', color='green')
plt.plot(tot_ret.index, tot_ret['ETF[Cumulative]'], label='ETF Cumulative Returns', color='red')
plt.title('Cumulative Portfolio Return vs ETF')
plt.legend()
plt.show()
