# 🚗 Geo Accident ML Dashboard

An interactive machine learning dashboard for analysing and predicting road accident severity across the United States. Built with Streamlit, scikit-learn, XGBoost, and SHAP.

---

## 📸 Features

- **Interactive dashboard** — select features, target, and problem type directly in the browser
- **Multiple ML models** — Random Forest, XGBoost, and Neural Network (MLP) for both classification and regression
- **Automatic preprocessing** — drops irrelevant columns, encodes categorical features, fills missing values
- **Spatial clustering** — DBSCAN geo-clustering of accident locations
- **Time feature engineering** — hour, month, day of week, rush hour flag, cyclic encoding
- **Rich visualisations:**
  - Severity distribution bar chart
  - Accident frequency heatmap (hour × day of week)
  - Feature correlation matrix
  - Feature importance bar charts (per model)
  - Confusion matrices (classification)
  - Actual vs Predicted scatter plots (regression)
  - SHAP beeswarm plot for the best model
- **Data drift detection** — KS-test between train and test distributions
- **Model persistence** — save and reload trained models with joblib

---

## 🗂️ Project Structure

```
Geo-accident ML/
├── dashboard/
│   ├── __init__.py
│   └── app.py               # Streamlit application entry point
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Data loading, cleaning, encoding, splitting
│   ├── feature_engineering.py  # Time features, rush hour flag, cyclic encoding
│   ├── spatial.py           # DBSCAN spatial clustering
│   ├── models.py            # Model registry, training loop, save/load
│   ├── evaluation.py        # Metrics + all plot functions
│   └── drift.py             # KS-test drift detection
├── utils/
│   ├── __init__.py
│   ├── config.py            # All project constants and paths
│   └── logger.py            # Colour-coded logger with rotating file handler
├── data/
│   └── accidents.csv        # Dataset (not included — see below)
├── models/                  # Saved model files (auto-created)
├── logs/                    # Log files (auto-created)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

This project uses the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle.

1. Download `US_Accidents_March23.csv` (or any version) from Kaggle
2. Rename it to `accidents.csv`
3. Place it in the `data/` folder

> The dataset is not included in this repository due to its size (~1 GB).

---

## 🚀 Getting Started


### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Place `accidents.csv` in the `data/` folder (see Dataset section above).

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🧠 Models

| Model | Classification | Regression |
|---|---|---|
| Random Forest | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| Neural Network (MLP) | ✅ | ✅ |

All models are trained locally on your machine. Training progress is shown live in the terminal with an ASCII progress bar and elapsed time per model.

---

## ⚙️ Configuration

All project settings are in `utils/config.py`:

| Setting | Default | Description |
|---|---|---|
| `TEST_SIZE` | `0.2` | Fraction of data used for testing |
| `RANDOM_STATE` | `42` | Random seed for reproducibility |
| `DRIFT_PVALUE_THRESHOLD` | `0.05` | KS-test p-value threshold for drift detection |
| `DBSCAN_EPS` | `0.05` | DBSCAN neighbourhood radius (~5.5 km) |
| `DBSCAN_MIN_SAMPLES` | `50` | Minimum points to form a DBSCAN cluster |

---

## 📊 Dashboard Walkthrough

1. **Data Preview** — inspect the first 100 rows after cleaning and encoding
2. **Exploratory Analysis** — severity distribution and accident heatmap (visible immediately)
3. **Model Configuration** — pick features, target column, and problem type
4. **Correlation Heatmap** — updates live as you select features
5. **Train Models** — click the button; progress prints in the terminal
6. **Evaluation Results** — metrics table with best highlighted in green, worst in red
7. **Feature Importance** — bar chart per model
8. **Confusion Matrices / Actual vs Predicted** — depending on problem type
9. **SHAP Beeswarm** — explainability for the best tree-based model
10. **Drift Detection** — enable in sidebar, then select a feature to inspect

---

## 📁 Logs

All logs are written to `logs/app.log` with rotation. The console output is colour-coded by log level.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — dashboard UI
- [scikit-learn](https://scikit-learn.org/) — Random Forest, MLP, preprocessing
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting models
- [SHAP](https://shap.readthedocs.io/) — model explainability
- [pandas](https://pandas.pydata.org/) — data manipulation
- [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) — visualisations
- [scipy](https://scipy.org/) — statistical drift detection

---

