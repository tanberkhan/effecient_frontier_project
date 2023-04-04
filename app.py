import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pytz
import seaborn as sns

import scipy.optimize as sco
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()

class Portfolio:


    def __init__(self, assets):
        self.assets = assets
        self.mean_returns = None
        self.cov_matrix = None
        self.num_assets = len(assets)
        self.weights = np.random.random(self.num_assets)
        self.weights /= np.sum(self.weights)


    def download_data(self, start_date, end_date):
        yf.pdr_override()

        data=pdr.get_data_yahoo(self.assets, start=start_date, end=end_date)

        print(data)
        self.daily_returns = data['Adj Close'].pct_change()
        self.mean_returns = self.daily_returns.mean()

        print(self.mean_returns)

        self.cov_matrix = self.daily_returns.cov().to_numpy()


    def calculate_portfolio_performance(self, weights):
        portfolio_return = np.dot(self.mean_returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        return -sharpe_ratio


    def generate_efficient_frontier(self, num_portfolios):
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            portfolio_return = np.dot(self.mean_returns, weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
        return results


    def optimize_portfolio(self):
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for i in range(self.num_assets))
        initial_weights = self.num_assets * [1. / self.num_assets]
        optimal_weights = sco.minimize(self.calculate_portfolio_performance, initial_weights, method='SLSQP',
                                       bounds=bounds, constraints=constraints)
        return optimal_weights.x


# yf.pdr_override()

# Set up the assets
assets = ['AAPL', 'GOOG', 'AMZN',  'NFLX', 'TSLA', 'NVDA','SPY']
portfolio = Portfolio(assets)

# # Download the historical data

start_date='2020-10-24', 
end_date='2022-10-24'
# start_date = '2019-03-14'
# end_date = '2021-03-14'
portfolio.download_data( '2020-10-24', '2022-10-24')

# Calculate the mean, variance, and correlation matrix for all assets
mean_returns = portfolio.mean_returns
cov_matrix = portfolio.cov_matrix
corr_matrix = portfolio.daily_returns.corr()

# Calculate the efficient frontier and optimal weights for the portfolio
num_portfolios = 5000
results = portfolio.generate_efficient_frontier(num_portfolios)
optimal_weights = portfolio.optimize_portfolio()

# Graph the results
sns.set()
plt.figure(figsize=(10, 7))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(results[1, :].min(), results[0, :].max(), marker='*', s=500, c='r', label='Optimal Portfolio')
plt.legend()
plt.title('Efficient Frontier')
plt.show()

print('Optimal weights:', optimal_weights)
