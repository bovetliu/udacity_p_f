# sharpe ratio
# (portfolio_return - risk_free_rate_of_return) / (std dev of portfolio return)

# SR = sqrt(252) * mean(daily_returns - daily_risk_free_return) / std(daily_returns)

"""
important portfolio statistics:
1. Cumulative Return
2. Average Daily Return
3. Risk (Std)
4. Sharpe Ratio

"""

from math import sqrt

if __name__ == "__main__":
    print(sqrt(252.0) * (0.001 - 0.0002) / 0.001)
