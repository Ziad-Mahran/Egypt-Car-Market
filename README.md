# 🚗 Egyptian Used Car Price Estimator & Market Categories

A machine-learning project that estimates a fair price for a used car in Egypt , suggest listings for buyers and groups the market into segments. It covers the whole workflow: data collection, cleaning, EDA, preprocessing pipelines, model training and tuning, clustering, and a bilingual (English/Arabic) Streamlit app.

**Live demo:** https://egypt-car-market.streamlit.app/

## What it does

- **Price Estimator:** enter brand, model, year, mileage and other details to get an estimated price, a likely range, and the closest real listings, each labelled as below, close to, or above the estimate.
- **Market Segments:** explore three market groups found with K-Means, filtered by budget, brand and condition.
- **Bilingual interface:** English and Arabic (right-to-left) from a sidebar switch.

Only classical machine learning is used (scikit-learn and XGBoost), with no deep learning.

## Results at a glance

Test set (30% of the data, never used for fitting or tuning). Prices are in EGP.

| Model | MAE | Mean % error | Median % error | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 288,250 | 22.2% | 12.8% | 0.655 |
| Ridge | 288,589 | 22.2% | 12.8% | 0.655 |
| Lasso | 389,304 | 27.2% | 18.5% | 0.553 |
| Random Forest (baseline) | 446,205 | 32.9% | 23.0% | 0.479 |
| **Random Forest (tuned)** | **215,042** | **17.0%** | **8.9%** | **0.748** |
| XGBoost (tuned), deployed | 236,228 | 18.5% | 10.6% | 0.727 |

The tuned random forest scores best on the test set, but it fits the training data far more closely than the test data (5.6% mean error on train vs. 17.0% on test). XGBoost shows a smaller gap (15.7% vs. 18.5%), so it is the model used in the app.

---

## Project workflow

### 1. Data collection

Listings were collected from Egyptian used-car marketplaces:

- **Dubizzle** listings were scraped with a Python script (`scrape_dubizzle_egypt_cars.py`, with checkpointing so a long run can resume) and ReefAPI.
- A public **Kaggle** dataset of Egyptian used cars was added to widen coverage.


Both were merged in `DataConcat.ipynb` after fixing noise and standardizing column names. The merged file has **37,391 rows and 14 columns** and keeps a `source` column (`dubizzle`, `hatla2ee`, `contactcars`).

| Column | Description |
|---|---|
| `listing_id`, `url` | Identifier and link to the original listing |
| `make`, `model` | Brand and model |
| `year`, `kilometers` | Manufacturing year and mileage |
| `price` | Asking price in EGP (the target) |
| `transmission`, `fuel_type`, `body_type`, `exterior_color`, `condition` | Car attributes |
| `payment_options` | Payment methods (about 73% missing, dropped later) |
| `source` | Where the listing came from |

### 2. Cleaning (`Egypt_Used_Cars.ipynb`)

1. **Drop the index column** left over from the CSV export.
2. **Remove duplicates.** There were no exact duplicates. 1,838 near-duplicates (identical on every field except `listing_id`, `url`, `source` and `payment_options`) were removed, which also catches the same car posted on more than one site.
3. **Inspect every column** for noise and missing values: 113 brands, 1,362 models, and heavy missingness in `body_type` and `exterior_color` (about 16.5k rows each).
4. **Visualize before cleaning:** histograms, box plots, and price vs. year and mileage scatter plots. Price and mileage are strongly right-skewed.
5. **Create `age`** (`2027 - year`), then remove rows with impossible mileage per year using a data-driven cap (Q3 + 3×IQR, about 50,130 km/year). Rows with unknown mileage are kept and imputed later.
6. **Check outliers with IQR** on price and mileage. These were inspected, not deleted automatically, because many "outliers" are genuine luxury cars or very old cars.
7. **Fix mislabeled condition.** Used cars aged 1 year or less with under 1,000 km (31 candidates) were compared with the median price of genuine new cars of the same model (median ratio about 0.96), and relabeled as **New**. Remaining used cars with 0 km were set to missing, since 0 means unknown here.
8. **Drop `year`** (replaced by `age`) and **`payment_options`**.

About **35.4k rows** remain.

### 3. Train/test split and train-only price filter

- 70/30 split with `random_state=11`.
- A **price filter is computed on the training set only**: for every make/model with at least 20 listings, the log-ratio of each price to its model's median is calculated, and rows outside Q1 − 3×IQR and Q3 + 3×IQR (0.29× to 3.47× the model median) are dropped. This removed **402 training rows** and avoids using test data to make cleaning decisions.

### 4. Preprocessing pipeline

Everything is inside one scikit-learn `ColumnTransformer`, fitted only on training data:

| Features | Steps |
|---|---|
| `age` | Median imputation, `PowerTransformer` |
| `kilometers` | Median imputation, `log1p`, `StandardScaler` |
| `make`, `model`, `condition`, `fuel_type`, `transmission` | Most-frequent imputation, one-hot encoding (rare values below 20 listings grouped together) |
| `body_type`, `exterior_color` | Missing values filled with `"Missing"` (they are about 46% missing, so missingness is kept as information), one-hot encoding |

The target is modeled as `log1p(price)` using `TransformedTargetRegressor`, and predictions are converted back to EGP automatically.

### 5. Modeling

All models share the same preprocessing and log-target setup.

- **Linear Regression** as the baseline.
- **Ridge** (`RidgeCV`, alphas from 0.01 to 1000) and **Lasso** (`LassoCV`) as regularized versions.
- **Random Forest:** a baseline, then tuned with `RandomizedSearchCV` (20 candidates, 5-fold CV, scored by root mean squared log error). Best settings: `max_features=0.7`, `max_depth=None`, `min_samples_leaf=1`, `min_samples_split=2`.
- **XGBoost:** tuned the same way. Best settings: `n_estimators=500`, `max_depth=9`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.6`, `reg_alpha=0.1`, `reg_lambda=5`, `min_child_weight=8`, `gamma=0.3`.

Metrics reported: MAE, RMSE, mean % error (MAPE), median % error, and R².

### 6. Market segmentation (K-Means)

K-Means runs on the same preprocessed features as the price model (price itself is **not** a feature). K from 2 to 8 was evaluated with the elbow method and silhouette score. The silhouette peaks at K=2 (0.376), while the elbow suggests K=3, which was chosen for more useful segments.

| Group | Listings | Median price (EGP) | Mean age (years) |
|---|---:|---:|---:|
| Older and budget | 13,455 | 270,000 | 24.0 |
| Mid-age | 19,551 | 1,080,000 | 7.1 |
| Nearly new | 2,388 | 1,727,500 | 2.2 |

The final K=3 clustering has a silhouette of 0.16 and a Davies-Bouldin index of 1.89, so the groups overlap. They separate mainly by age and mileage.

### 7. Deployment

- The best XGBoost pipeline (preprocessing and model together) is saved with `joblib` as `xgboost_car_price_model.pkl`.
- The Streamlit app has three pages: **Home**, **Price Estimator**, and **Market Segments**. The estimator builds its dropdowns from the dataset, so the values always match what the model was trained on, and it finds similar listings from the cleaned data with links to the originals.
- It is deployed on **Streamlit Community Cloud**.

---

## Repository structure

```
├── Home.py                       # Streamlit home page (run this)
├── utils.py                      # Shared settings, translations, data/model loading
├── pages/
│   ├── Price_Prediction.py       # Price Estimator page
│   └── Market_Clustering.py      # Market Segments page
├── Egypt_Used_Cars.ipynb         # Cleaning, EDA, modeling, clustering
├── DataConcat.ipynb              # Merging the data sources
├── scrape_dubizzle_egypt_cars.py # Dubizzle scraper
├── banner.jpg
├── xgboost_car_price_model.pkl   # Trained pipeline
├── requirements.txt
└── .streamlit/config.toml        # App theme
```

## Run it locally

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
streamlit run Home.py
```

The app expects `Egypt_Used_Cars_clean.csv` and `xgboost_car_price_model.pkl` in the project folder. Install the same `scikit-learn` and `xgboost` versions used for training, or the saved model may not load.

## Limitations

- Prices are **asking prices**, not final sale prices.
- The data is a **snapshot from several sources collected at different times**, so prices drift with inflation and market changes.
- Rare models have few listings, so their estimates are less reliable. The typical error is about ±11%, and single estimates can be further off.
- The clusters overlap (silhouette 0.16) and are best read as rough market tiers.



## Author

**Ziad Osama**, third-year AI engineering student, Faculty of Engineering.
