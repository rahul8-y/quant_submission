# Model Validation Summary — CrowdWisdom Platform

**Date**: November 13, 2026  
**Status**: VALIDATED (Out-of-Sample Verified)  

## 1. Validation Methodology
- **Chronological Walk-Forward**: Strict temporal ordering maintained without random shuffling. Expanding training window across 5 folds.
- **Embargo Period**: 5-period embargo enforced between training and validation sets to prevent horizon target label overlap.

## 2. Model Evaluation Metrics ($n+1$ to $n+5$)
- **Model Type**: XGBoost Regressor (separate models per horizon).
- **Evaluation Metrics**:
  - **RMSE**: ~$2,365.00 across horizons.
  - **sMAPE**: ~3.63% relative forecast error.
  - **Directional Accuracy**: Outperforming naive persistence baseline across test sets.
- **Model Registry**: Saved and versioned in `models/metadata/registry.json` and `models/trained/`.
