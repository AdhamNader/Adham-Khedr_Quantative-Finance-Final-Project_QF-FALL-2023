import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import cvxpy as cp

# Define the ETF and placeholder assets
etf_ticker = 'SPY'  # Example ETF
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BRK-B', 'JNJ', 'V', 'PG', 'JPM']  # Example assets

# Step 2: Retrieve Historical Data
start_date = '2018-01-01'
end_date = '2023-01-01'
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
etf_data = yf.download(etf_ticker, start=start_date, end=end_date)['Adj Close']

# Step 3: Calculate Price Momentum Factors (e.g., 12-month rate of change)
momentum_factor = data.pct_change(periods=252).dropna()

# Step 4: Calculate Z-Factor Scores
z_scores = momentum_factor.apply(zscore)

# Step 5: Identify Long and Short Baskets
long_basket = z_scores.iloc[-1].nlargest(10).index.tolist()
short_basket = z_scores.iloc[-1].nsmallest(10).index.tolist()


# Function to calculate monthly returns for a given basket
def calculate_monthly_returns(data, basket):
    monthly_data = data[basket].resample('M').last()
    monthly_returns = monthly_data.pct_change().dropna()
    return monthly_returns.mean(axis=1)

# Function to calculate optimized weights
def optimize_weights(returns):
    cov_matrix = returns.cov()
    n = len(returns.columns)
    
    # Define optimization variables
    weights = cp.Variable(n)
    
    # Define objective (maximize return / risk)
    portfolio_return = cp.sum(returns.mean() @ weights)
    portfolio_risk = cp.quad_form(weights, cov_matrix)
    objective = cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))
    
    # Constraints
    constraints = [cp.sum(weights) == 0, cp.norm(weights, 1) <= 1]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value

# Step 6: Backtest the Algorithm
long_returns = calculate_monthly_returns(data, long_basket)
short_returns = calculate_monthly_returns(data, short_basket)

# Calculate net portfolio returns (assuming equal weighting and short selling)
net_returns = long_returns - short_returns

# Cumulative returns for the portfolio
cumulative_returns = (1 + net_returns).cumprod() - 1

# Calculate ETF monthly returns for comparison
etf_monthly_returns = etf_data.resample('M').last().pct_change().dropna()
etf_cumulative_returns = (1 + etf_monthly_returns).cumprod() - 1

# Step 7: Make Charts for the Results
plt.figure(figsize=(20, 5))
plt.bar(net_returns.index, net_returns, label='Portfolio', color='blue')
plt.bar(etf_monthly_returns.index, etf_monthly_returns, label='ETF', color='red')
plt.title('Monthly Portfolio Returns vs ETF')
plt.legend()
plt.show()

plt.figure(figsize=(18, 4))
plt.plot(long_returns.index, long_returns, label='Long Picks', color='blue')
plt.plot(short_returns.index, short_returns, label='Short Picks', color='yellow')
plt.plot(etf_monthly_returns.index, etf_monthly_returns, label='ETF', color='red')
plt.title('Monthly Return: Long Picks vs Short Picks vs ETF')
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio', color='green')
plt.plot(etf_cumulative_returns.index, etf_cumulative_returns, label='SPY', color='red')
plt.title('Cumulative Portfolio Return vs SPY')
plt.legend()
plt.show()

# Extra Credit: Optimized Weighting
monthly_returns = data.resample('M').last().pct_change().dropna()
optimized_weights = monthly_returns.apply(optimize_weights, axis=1)

print("Long Basket:", long_basket)
print("Short Basket:", short_basket)
print("\nOptimized Weights:\n", optimized_weights)