# Content Monetization Modeler

Predicts YouTube ad revenue (USD) for a video-day using Linear Regression,
compared against four other models, with a Streamlit app.

## Results (20% test set)
| Model | R2 | RMSE | MAE |
|---|---|---|---|
| Lasso Regression | 0.9526 | 13.48 | 3.08 |
| Linear Regression (final) | 0.9526 | 13.48 | 3.15 |
| Ridge Regression | 0.9526 | 13.48 | 3.16 |
| Gradient Boosting | 0.9522 | 13.54 | 3.71 |
| Random Forest | 0.9517 | 13.60 | 3.80 |

5-fold CV R2 for Linear Regression: 0.9497 +/- 0.0015.

## Key insights
- Watch time is the dominant driver of revenue (correlation about 0.99).
- Category, device and country change average revenue by only about $1-2.
- The relationship is essentially linear, so complex models did not beat Linear Regression.
- With watch time present: R2 = 0.9988, MAE = $0.69. With watch time missing
  (about 5% of rows, median-imputed): R2 = 0.02, MAE = $52.58. These rows cause
  most of the RMSE.
- The data is synthetic, so real revenue also depends on factors not in this dataset.

## Project structure
- `Data/` raw dataset and cleaned CSV
- `Models/` saved model and comparison table
- `notebooks/` analysis notebook
- `app.py` Streamlit app

## How to run
    pip install -r requirements.txt
    python -m streamlit run app.py

## Method
1. Removed 2,400 duplicate rows (122,400 to 120,000).
2. Imputed missing likes, comments and watch time with the median, inside the model pipeline (no data leakage).
3. IQR check found no impossible values, so all rows were kept.
4. Added features: engagement_rate, like_rate, comment_rate, watch_time_per_view, month, day_of_week.
5. One-hot encoding for categories, standard scaling for numbers.
6. Trained Linear, Ridge, Lasso, Random Forest and Gradient Boosting; compared R2, RMSE, MAE.
