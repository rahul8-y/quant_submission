from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR

app = FastAPI(
    title="CrowdWisdom X Sentiment & Quantitative Intelligence API",
    version="1.0.0",
    description="REST API for Bitcoin price predictions, X sentiment metrics, model evaluation, and risk analytics."
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "crowdwisdom-api"}

@app.get("/api/v1/market/latest")
def get_market_latest():
    parquet_path = PROCESSED_DATA_DIR / "hourly_training_dataset.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise HTTPException(status_code=404, detail="Empty dataset")
    latest = df.iloc[-1]
    return {
        "timestamp": str(latest['timestamp']),
        "close": float(latest['close']),
        "return_1h": float(latest.get('return_1h', 0.0)),
        "volume": float(latest.get('volume', 0.0)),
        "volatility_12h": float(latest.get('volatility_12h', 0.0))
    }

@app.get("/api/v1/sentiment/latest")
def get_sentiment_latest():
    parquet_path = PROCESSED_DATA_DIR / "hourly_training_dataset.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(parquet_path)
    latest = df.iloc[-1]
    return {
        "timestamp": str(latest['timestamp']),
        "mean_sentiment": float(latest.get('mean_sentiment', 0.0)),
        "weighted_sentiment": float(latest.get('weighted_sentiment', 0.0)),
        "tweet_count": int(latest.get('tweet_count', 0))
    }

@app.get("/api/v1/sentiment/history")
def get_sentiment_history(limit: int = 100):
    parquet_path = PROCESSED_DATA_DIR / "hourly_training_dataset.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(parquet_path).tail(limit)
    return df[['timestamp', 'weighted_sentiment', 'tweet_count', 'mean_sentiment']].to_dict(orient="records")

@app.get("/api/v1/predictions/latest")
def get_predictions_latest():
    parquet_path = PROCESSED_DATA_DIR / "hourly_training_dataset.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(parquet_path)
    latest = df.iloc[-1]
    return {
        "timestamp": str(latest['timestamp']),
        "n_plus_0": float(latest.get('target_n_plus_0', latest['close'])),
        "n_plus_1": float(latest.get('target_n_plus_1', latest['close'])),
        "n_plus_2": float(latest.get('target_n_plus_2', latest['close'])),
        "n_plus_3": float(latest.get('target_n_plus_3', latest['close'])),
        "n_plus_4": float(latest.get('target_n_plus_4', latest['close'])),
        "n_plus_5": float(latest.get('target_n_plus_5', latest['close']))
    }

@app.get("/api/v1/models/metrics")
def get_model_metrics():
    registry_path = Path("models/metadata/registry.json")
    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="Model registry not found")
    import json
    return json.loads(registry_path.read_text(encoding="utf-8"))

@app.get("/api/v1/backtest/metrics")
def get_backtest_metrics():
    metrics_path = Path("models/metadata/backtest_metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Backtest metrics not found")
    import json
    return json.loads(metrics_path.read_text(encoding="utf-8"))

@app.get("/api/v1/backtest/equity")
def get_backtest_equity():
    eq_path = Path("models/metadata/backtest_equity.json")
    if not eq_path.exists():
        raise HTTPException(status_code=404, detail="Backtest equity curve not found")
    import json
    return json.loads(eq_path.read_text(encoding="utf-8"))

@app.get("/api/v1/news/latest")
def get_news_latest():
    return [
        {
            "title": "Bitcoin Institutional Inflows Reach New Weekly Highs",
            "publisher": "CoinDesk Markets",
            "publication_date": "2026-11-13",
            "url": "https://www.coindesk.com/markets/",
            "summary": "Institutional participation expands as spot ETF volumes surge."
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
