# CreditLens

A small-business loan default predictor built on ~900k real SBA loans. An XGBoost model is wrapped in a Streamlit app that returns a risk score for any loan you describe, along with a SHAP breakdown of which features pushed the score up or down.

## Results

Trained on 717,733 loans and evaluated on 179,434 held-out loans. The historical default rate in the data is 17.6%.

| Metric            | Score |
| ----------------- | ----- |
| ROC-AUC           | 0.976 |
| Average precision | 0.904 |

These numbers are written to `models/model_stats.json` at the end of training and shown in the app's sidebar.

## Quick start

1. Clone the repo and `cd` into it.
2. Create a virtual environment and install the dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate         # Windows
   source .venv/bin/activate      # macOS / Linux
   pip install -r requirements.txt
   ```
3. Download `SBAnational.csv` from [Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied) and drop it into `data/`.
4. Train the model by running the three notebooks in order. Each one writes its artifacts to `data/` or `models/`.
   ```
   jupyter notebook notebooks/01_data_exploration.ipynb
   jupyter notebook notebooks/02_feature_engineering.ipynb
   jupyter notebook notebooks/03_model_training.ipynb
   ```
5. Launch the app:
   ```
   streamlit run app/streamlit_app.py
   ```

## How it works

The pipeline is three stages connected by files on disk:

1. **Exploration** (`01_data_exploration.ipynb`). Sanity checks on the raw CSV: missing data, target balance, default rates by industry and state.
2. **Feature engineering** (`02_feature_engineering.ipynb`). Drops leakage columns (anything that only exists after a loan defaults), parses dollar strings to floats, encodes categoricals, and builds ratio features such as the SBA guarantee share and loan-per-job. Writes `X_processed.csv`, `y_processed.csv`, `feature_names.json`, and `state_mapping.json`.
3. **Training** (`03_model_training.ipynb`). 80/20 stratified split, class imbalance handled with `scale_pos_weight` instead of resampling, fits an XGBoost classifier, then computes SHAP values on a 5k-row sample. Saves the model, the explainer, and the stats file.

The Streamlit app loads those five artifacts at startup. When you submit a loan it reconstructs the same feature vector the training pipeline produced, scores it, and renders the per-feature SHAP contributions next to the score.

## Layout

```
credit-lens/
├── app/
│   └── streamlit_app.py
├── data/                # raw and processed data (gitignored)
├── models/              # trained .pkl files and JSON metadata
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── requirements.txt
└── README.md
```

## About the data

`SBAnational.csv` is the SBA's national loan record (~887k rows, ~180MB). Each row is one approved loan, and the outcome is encoded in `MIS_Status`: `P I F` for paid in full, `CHGOFF` for charged off.

A few things worth knowing before you run the notebooks:

- Dollar columns are stored as strings like `$12,345.00` and need to be parsed.
- `ChgOffDate`, `ChgOffPrinGr`, and `BalanceGross` only exist for defaulted loans, so using them as features would leak the answer. The feature-engineering step drops them.
- `ApprovalFY` contains at least one non-numeric value (`'1976A'`) that has to be coerced.
- Defaults make up roughly 18% of the data rather than 50%, so the training step uses `scale_pos_weight` to keep the model from biasing toward the majority class.

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, joblib. Plots via matplotlib and seaborn. Tested on Python 3.14.
