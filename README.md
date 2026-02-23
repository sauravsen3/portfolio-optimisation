# 📈 Portfolio Optimisation — Markowitz Efficient Frontier

A Python implementation of Modern Portfolio Theory that finds the optimal allocation of assets to maximise risk-adjusted returns, visualised through the Efficient Frontier.

## About Me

I am a Mechatronics Engineering graduate and a postgraduate in Finance from Henley Business School, looking to enter the field of finance in data science, machine learning, or quantitative finance.

- 💼 LinkedIn: [linkedin.com/in/sauravsen34](https://www.linkedin.com/in/sauravsen34)
- 📧 Email: saurav0sen34@gmail.com

---

## What Does This Project Do?

Given a set of assets, this project downloads historical price data, calculates returns, simulates 5000 random portfolio weight combinations, and finds the optimal portfolio — the one that maximises the Sharpe Ratio. The result is plotted as the Efficient Frontier.

---

## Key Concepts

**Sharpe Ratio** — the risk-to-reward ratio of a portfolio. Calculated as annual return divided by annual volatility. A higher Sharpe Ratio means more return per unit of risk taken. It is the primary metric used to compare portfolio performance.

```
Sharpe Ratio = Annual Return / Annual Volatility
```

**Efficient Frontier** — the curve representing portfolios that minimise risk for a given level of return, or maximise return for a given level of risk. Every point below the curve is suboptimal. The red star marks the optimal portfolio — the single point with the highest Sharpe Ratio.

**Markowitz Insight** — combining assets that don't move together (low correlation) reduces portfolio risk without sacrificing return. This is the mathematical proof of diversification.

---

## How It Works

```
get_data() → portfolio_performance() → simulate_portfolios() → get_optimal_portfolio() → plot
```

1. **get_data()** — downloads 5 years of daily close prices for selected tickers via yfinance and calculates daily returns

2. **portfolio_performance()** — calculates annualised return, volatility, and Sharpe Ratio for any set of weights using matrix multiplication on the covariance matrix

3. **simulate_portfolios()** — generates 5000 random weight combinations, normalised to sum to 1, and records the return/volatility/Sharpe for each

4. **get_optimal_portfolio()** — uses scipy `minimize` on the negative Sharpe Ratio (minimising negative = maximising positive) with constraints that weights sum to 1 and no short selling

---

## Output

```
--- Optimal Portfolio ---
AAPL: 67.3%
GLD:  24.1%
AGG:   8.6%
Expected Annual Return: 24.5%
Expected Annual Volatility: 18.2%
Sharpe Ratio: 1.34
```

A scatter plot is saved as `efficient_frontier.png` — each dot is one simulated portfolio, coloured by Sharpe Ratio. The red star marks the mathematically optimal allocation.

---

## Important Caveat

The optimal weights are based purely on historical data. A model suggesting 80% in a single stock should be treated as one input alongside fundamental analysis, macro views, and risk limits — not as a standalone instruction. Past performance is not indicative of future results.

---

## How To Run

```bash
git clone https://github.com/sauravsen3/portfolio-optimisation.git
cd portfolio-optimisation
pip install -r requirements.txt
python portfolio.py
```

---

## Project Structure

```
portfolio-optimisation/
│
├── portfolio.py              # Core logic — simulation and optimisation
├── requirements.txt          # Python dependencies
└── efficient_frontier.png    # Generated on first run
```

---

## Tech Stack

- **yfinance** — historical price data
- **pandas / numpy** — returns calculation and matrix operations
- **scipy** — constrained optimisation to find maximum Sharpe Ratio
- **matplotlib** — Efficient Frontier scatter plot

---

*Part of a series of quantitative finance projects. Previous: Financial News Sentiment Analysis. Next: Algorithmic Trading Backtest.*
