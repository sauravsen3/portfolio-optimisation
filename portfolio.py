import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize


def get_data(tickers, start, end):
    """
    Download adjusted close prices for multiple tickers.
    Returns DataFrame of daily returns.
    """
    data = yf.download(tickers, start=start, end=end)['Close']
    returns = data.pct_change().dropna()
    print(f"Downloaded {len(returns)} days of data for {tickers}")
    return returns


def portfolio_performance(weights, returns):
    """
    Calculate annualised return, volatility and Sharpe Ratio.
    """
    annual_return = np.sum(returns.mean() * weights) * 252
    annual_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )
    sharpe_ratio = annual_return / annual_volatility
    return annual_return, annual_volatility, sharpe_ratio


def simulate_portfolios(returns, num_simulations=5000):
    """
    Simulate random portfolio weight combinations.
    Returns arrays of results for plotting the Efficient Frontier.
    """
    num_assets = len(returns.columns)
    results = np.zeros((3, num_simulations))
    weights_record = []

    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)

        ret, vol, sharpe = portfolio_performance(weights, returns)
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
        weights_record.append(weights)

    return results, weights_record


def get_optimal_portfolio(returns):
    """
    Use scipy minimise to find weights that maximise Sharpe Ratio.
    """
    num_assets = len(returns.columns)

    def negative_sharpe(weights):
        _, _, sharpe = portfolio_performance(weights, returns)
        return -sharpe

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1 / num_assets]

    result = minimize(negative_sharpe, initial_weights,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x


def plot_efficient_frontier(results, optimal_weights, returns, tickers):
    """
    Plot the Efficient Frontier with optimal portfolio highlighted.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    scatter = ax.scatter(results[1], results[0],
                         c=results[2], cmap='viridis',
                         alpha=0.5, s=10)
    plt.colorbar(scatter, label='Sharpe Ratio')

    opt_ret, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, returns)
    ax.scatter(opt_vol, opt_ret, color='red', marker='*', s=300,
               label=f'Optimal Portfolio (Sharpe: {opt_sharpe:.2f})')

    ax.set_xlabel('Annual Volatility')
    ax.set_ylabel('Annual Return')
    ax.set_title('Efficient Frontier — Portfolio Optimisation')
    ax.legend()
    plt.tight_layout()
    plt.savefig('efficient_frontier.png')
    plt.show()


if __name__ == "__main__":
    tickers = ['AAPL', 'GLD', 'AGG']
    start = datetime.now() - timedelta(days=365*5)
    end = datetime.now()

    returns = get_data(tickers, start, end)
    results, weights_record = simulate_portfolios(returns)
    optimal_weights = get_optimal_portfolio(returns)

    opt_ret, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, returns)

    print("\n--- Optimal Portfolio ---")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.1%}")
    print(f"Expected Annual Return: {opt_ret:.2%}")
    print(f"Expected Annual Volatility: {opt_vol:.2%}")
    print(f"Sharpe Ratio: {opt_sharpe:.2f}")

    plot_efficient_frontier(results, optimal_weights, returns, tickers)
