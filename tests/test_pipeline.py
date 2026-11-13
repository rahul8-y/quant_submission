import pytest
import pandas as pd
import numpy as np
from src.processing.cleaner import clean_tweets, aggregate_sentiment_by_hour
from src.features.feature_engineering import compute_technical_indicators
from src.sentiment.sentiment_engine import run_sentiment_engine

def test_clean_tweets():
    df = pd.DataFrame({
        'date': ['2026-02-10 23:59:04', '2026-02-10 23:58:48'],
        'text': ['Bitcoin is going to the moon! https://t.co/xyz', 'Sell BTC now #crypto'],
        'user_followers': [1000, 500],
        'is_retweet': [False, True]
    })
    cleaned = clean_tweets(df)
    assert not cleaned.empty
    assert 'clean_text' in cleaned.columns
    assert 'hour_bucket' in cleaned.columns

def test_sentiment_engine():
    df = pd.DataFrame({
        'clean_text': ['Bitcoin to the moon! Buy buy buy', 'Absolute crash dump sell'],
        'user_followers': [1000, 500]
    })
    scored = run_sentiment_engine(df)
    assert 'sentiment_score' in scored.columns
    assert 'weighted_sentiment_score' in scored.columns

def test_feature_engineering():
    dates = pd.date_range(start='2026-01-01', periods=50, freq='h', tz='UTC')
    prices = pd.DataFrame({
        'timestamp': dates,
        'open': np.linspace(50000, 51000, 50),
        'high': np.linspace(50200, 51200, 50),
        'low': np.linspace(49800, 50800, 50),
        'close': np.linspace(50100, 51100, 50),
        'volume': np.random.uniform(10, 100, 50),
        'tweet_count': np.random.randint(0, 50, 50),
        'mean_sentiment': np.random.uniform(-0.5, 0.5, 50),
        'weighted_sentiment': np.random.uniform(-1, 1, 50),
        'total_engagement': np.random.uniform(100, 1000, 50)
    })
    featured = compute_technical_indicators(prices)
    assert not featured.empty
    assert 'return_1h' in featured.columns
    assert 'rsi_14' in featured.columns
    assert 'target_n1' in featured.columns
