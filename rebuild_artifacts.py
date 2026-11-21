"""Rebuild Models/ and Data/ files for the Streamlit app from the raw Excel dataset."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR = BASE / "Data", BASE / "Models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 1. Find the raw Excel file
xlsx = DATA_DIR / "youtube_ad_revenue_dataset.xlsx"
if not xlsx.exists():
    home = Path.home()
    for root in [home / "OneDrive", home / "Desktop", home / "Documents", home / "Downloads"]:
        if root.exists():
            hits = list(root.rglob("youtube_ad_revenue_dataset*.xlsx"))
            if hits:
                xlsx = hits[0]
                break
print("Reading:", xlsx)
raw = pd.read_excel(xlsx)

# 2. Clean: remove duplicates
df = raw.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(df))


def add_features(d):
    d = d.copy()
    d["engagement_rate"] = (d["likes"] + d["comments"]) / d["views"]
    d["like_rate"] = d["likes"] / d["views"]
    d["comment_rate"] = d["comments"] / d["views"]
    d["watch_time_per_view"] = d["watch_time_minutes"] / d["views"]
    d["month"] = d["date"].dt.month
    d["day_of_week"] = d["date"].dt.dayofweek
    return d


NUMERIC = ["views", "likes", "comments", "watch_time_minutes", "video_length_minutes",
           "subscribers", "engagement_rate", "like_rate", "comment_rate",
           "watch_time_per_view", "month", "day_of_week"]
CATEGORICAL = ["category", "device", "country"]

# 3. Cleaned CSV for the app's charts (missing values filled with the median)
clean = df.copy()
for col in ["likes", "comments", "watch_time_minutes"]:
    clean[col] = clean[col].fillna(clean[col].median())
clean = add_features(clean)
clean.to_csv(DATA_DIR / "cleaned_youtube_ad_revenue.csv", index=False)

# 4. Train / test split (the pipeline fills missing values itself)
data = add_features(df)
X, y = data[NUMERIC + CATEGORICAL], data["ad_revenue_usd"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=10,
                                           n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                   random_state=42),
}
rows, fitted = [], {}
for name, est in models.items():
    print("Training", name, "...")
    pipe = Pipeline([("prep", prep), ("model", est)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted[name] = pipe
    rows.append({"model": name, "R2": r2_score(y_test, pred),
                 "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                 "MAE": mean_absolute_error(y_test, pred)})

results = pd.DataFrame(rows).sort_values("R2", ascending=False)
print(results.round(4).to_string(index=False))

# 5. Save the final Linear Regression and the comparison table
joblib.dump(fitted["Linear Regression"], MODEL_DIR / "revenue_model.pkl")
results.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
print("DONE. Files created:")
print(sorted(p.name for p in MODEL_DIR.iterdir()))
print(sorted(p.name for p in DATA_DIR.iterdir()))
