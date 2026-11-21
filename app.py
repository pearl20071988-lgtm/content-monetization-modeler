"""Streamlit app: YouTube Ad Revenue Predictor."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "revenue_model.pkl"
DATA_PATH = BASE_DIR / "Data" / "cleaned_youtube_ad_revenue.csv"
RESULTS_PATH = BASE_DIR / "Models" / "model_comparison.csv"

st.set_page_config(page_title="YouTube Revenue Predictor", layout="wide")
st.title("YouTube Ad Revenue Predictor")
st.caption("Linear Regression model trained on 120,000 cleaned video-day records.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


missing = [p for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error("These files were not found:\n\n" + "\n\n".join(str(p) for p in missing))
    st.stop()

model = load_model()
df = load_data()
med = df[["views", "likes", "comments", "watch_time_minutes",
          "video_length_minutes", "subscribers"]].median()


def build_row(views, likes, comments, watch_time, length, subscribers,
              category, device, country, report_date):
    d = pd.Timestamp(report_date)
    return pd.DataFrame([{
        "views": views, "likes": likes, "comments": comments,
        "watch_time_minutes": watch_time,
        "video_length_minutes": length, "subscribers": subscribers,
        "engagement_rate": (likes + comments) / views,
        "like_rate": likes / views,
        "comment_rate": comments / views,
        "watch_time_per_view": watch_time / views,
        "month": d.month, "day_of_week": d.dayofweek,
        "category": category, "device": device, "country": country,
    }])


tab_predict, tab_data, tab_model = st.tabs(
    ["Predict revenue", "Data insights", "Model performance"])

with tab_predict:
    st.subheader("Enter video details")
    c1, c2, c3 = st.columns(3)
    with c1:
        views = st.number_input("Views", min_value=1, value=int(med["views"]), step=100)
        likes = st.number_input("Likes", min_value=0, value=int(med["likes"]), step=50)
        comments = st.number_input("Comments", min_value=0, value=int(med["comments"]), step=10)
    with c2:
        watch_known = st.checkbox("I know the watch time", value=True)
        watch_time = st.number_input("Watch time (minutes)", min_value=0.0,
                                     value=float(med["watch_time_minutes"]),
                                     step=500.0, disabled=not watch_known)
        length = st.number_input("Video length (minutes)", min_value=0.5,
                                 value=float(med["video_length_minutes"]), step=0.5)
        subscribers = st.number_input("Channel subscribers", min_value=0,
                                      value=int(med["subscribers"]), step=10000)
    with c3:
        category = st.selectbox("Category", sorted(df["category"].unique()))
        device = st.selectbox("Device", sorted(df["device"].unique()))
        country = st.selectbox("Country", sorted(df["country"].unique()))
        report_date = st.date_input("Report date")

    if st.button("Predict ad revenue", type="primary"):
        if likes > views:
            st.warning("Likes cannot be greater than views. Please check your inputs.")
        else:
            try:
                wt = watch_time if watch_known else np.nan
                row = build_row(views, likes, comments, wt, length, subscribers,
                                category, device, country, report_date)
                prediction = max(0.0, float(model.predict(row)[0]))
                st.metric("Predicted ad revenue (USD)", f"${prediction:,.2f}")
                if not watch_known:
                    st.warning("Watch time is the strongest revenue driver. Without it "
                               "this estimate is rough (average error about $50).")
            except Exception as e:
                st.error(f"Could not make a prediction: {e}")

with tab_data:
    st.subheader("Cleaned dataset overview")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Mean revenue", f"${df['ad_revenue_usd'].mean():,.2f}")
    c.metric("Videos", f"{df['video_id'].nunique():,}")
    group_col = st.selectbox("Group by", ["category", "device", "country"])
    st.bar_chart(df.groupby(group_col)["ad_revenue_usd"].mean())
    st.scatter_chart(df.sample(3000, random_state=1),
                     x="watch_time_minutes", y="ad_revenue_usd")
    st.line_chart(df.groupby(df["date"].dt.to_period("M").astype(str))["ad_revenue_usd"].mean())

with tab_model:
    st.subheader("Comparison of the 5 models (test set)")
    if RESULTS_PATH.exists():
        st.dataframe(pd.read_csv(RESULTS_PATH).round(4), hide_index=True)
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["model"].coef_, index=names)
    st.bar_chart(coefs.sort_values(key=abs, ascending=False).head(8))
    st.caption("Bigger bars mean stronger effects. The two watch-time features dominate.")
