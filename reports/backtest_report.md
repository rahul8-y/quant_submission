# Quantitative Backtesting & Risk Analysis Report
## CrowdWisdom Quantitative Intelligence System

### 1. Executive Summary
This report presents the out-of-sample backtest results for the sentiment-augmented quantitative trading strategy on Bitcoin (BTC/USDT) hourly candles compared against a Buy-and-Hold benchmark.

### 2. Strategy Performance Metrics
```json
{
    "total_return_strategy": -0.22487175416703875,
    "total_return_buy_hold": -0.16322476601595926,
    "annualized_sharpe": -14.90447350116044,
    "annualized_sortino": -15.395389222399835,
    "max_drawdown": -0.23086936687774615,
    "number_of_trades": 175,
    "win_rate_pct": 20.090634441087612
}
```

### 3. Risk Management & Transaction Costs
- **Transaction Costs**: Modeled at 5 basis points (0.05%) per trade execution.
- **Risk-Free Rate**: Assumed at 0.00% for hourly Sharpe & Sortino annualization ($\sqrt{8760}$).
- **Maximum Drawdown**: Controlled via dynamic position sizing and flat neutral periods during high-volatility regimes.

---
*Generated automatically by CrowdWisdom Quantitative Intelligence System.*
