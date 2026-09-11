# Analysis environment

The final thesis-recorded analysis environment was:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- SciPy 1.17.0
- scikit-learn 1.8.0
- XGBoost 3.1.3
- SHAP 0.50.0
- Matplotlib 3.10.8
- joblib 1.5.3
- PyYAML 6.0.3

## Determinism / seed roles

- Primary split and tuning random state: `24101765`
- Five-fold paired model stability: `StratifiedGroupKFold`, `random_state=24101766`
- Repeated capacity validation seeds: `24101765` through `24101774`
- The primary final test partition remained fixed during operating-point selection.

## Reproducibility boundary

This repository currently includes documentation, aggregate results, figures and environment metadata. The original end-to-end `analysis_pipeline.py` and `analysis_config.yaml` referenced by the thesis reproducibility manifest are not currently available in this chat session and have not been recreated from memory. They should be added only from the exact original analysis directory after a restricted-data/path review.
