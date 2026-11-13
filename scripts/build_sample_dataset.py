import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.ingestion.load_data import fetch_binance_klines
from src.processing.cleaner import clean_tweets, aggregate_sentiment_by_hour
from src.features.feature_engineering import compute_technical_indicators
from src.models.train_models import train_multi_horizon_models
from src.backtesting.backtester import run_backtest
from src.reporting.generate_report import generate_evaluation_report
from config.settings import PROCESSED_DATA_DIR

def build():
    print("=== Building Sample/Processed Dataset & Running Pipeline ===")

    # 1. Load sample of tweets (first 20,000 rows for fast processing)
    csv_path = Path("bitcoin_tweets_latest.csv")
    if csv_path.exists():
        print(f"Reading sample from {csv_path}...")
        tweets_raw = pd.read_csv(csv_path, nrows=20000, low_memory=False, on_bad_lines='skip')
    else:
        tweets_raw = pd.DataFrame()

    # 2. Fetch Binance klines
    prices_raw = fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=720)
    if prices_raw.empty:
        # Create dummy price data if offline
        dates = pd.date_range(start='2026-01-01', periods=720, freq='h', tz='UTC')
        prices_raw = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(60000, 65000, 720) + np.random.normal(0, 100, 720),
            'high': np.linspace(60200, 65200, 720) + np.random.normal(0, 100, 720),
            'low': np.linspace(59800, 64800, 720) + np.random.normal(0, 100, 720),
            'close': np.linspace(60100, 65100, 720) + np.random.normal(0, 100, 720),
            'volume': np.random.uniform(100, 1000, 720)
        })

    # 3. Clean tweets & aggregate sentiment
    if not tweets_raw.empty:
        tweets_clean = clean_tweets(tweets_raw)
        merged_df = aggregate_sentiment_by_hour(tweets_clean, prices_raw)
    else:
        merged_df = prices_raw.copy()
        merged_df['tweet_count'] = np.random.randint(10, 100, len(merged_df))
        merged_df['mean_sentiment'] = np.random.uniform(-0.2, 0.3, len(merged_df))
        merged_df['weighted_sentiment'] = merged_df['mean_sentiment'] * 1.5
        merged_df['total_engagement'] = np.random.uniform(500, 5000, len(merged_df))

    # 4. Feature engineering
    featured_df = compute_technical_indicators(merged_df)

    processed_path = PROCESSED_DATA_DIR / "featured_data.csv"
    featured_df.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path} (Shape: {featured_df.shape})")

    # 5. Model training & backtesting
    models, metrics, test_df = train_multi_horizon_models(featured_df)

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'tweet_count', 'mean_sentiment', 'weighted_sentiment', 'total_engagement',
        'return_1h', 'return_4h', 'return_24h',
        'volatility_12h', 'volatility_24h', 'price_to_sma24',
        'rsi_14', 'sentiment_sma_6', 'sentiment_sma_24', 'tweet_count_sma_12'
    ]
    backtest_metrics = run_backtest(test_df, models, feature_cols)

    # 6. Generate report
    generate_evaluation_report(metrics, backtest_metrics)
    print("=== Build Completed Successfully! ===")

if __name__ == "__main__":
    build()
