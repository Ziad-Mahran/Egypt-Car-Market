import pandas as pd
import streamlit as st

from utils import TYPICAL_ERROR, load_data, sidebar, t

st.set_page_config(page_title="Egypt Used Cars", page_icon="🚗", layout="wide")
sidebar()
st.image("banner.jpg", use_container_width=True)
df = load_data()

st.title(t("app_title"))
st.write(t("app_subtitle"))

k1, k2, k3 = st.columns(3)
k1.metric(t("kpi_listings"), f"{len(df):,}")
k2.metric(t("kpi_median_price"), f"{df['price'].median():,.0f} {t('egp')}")
k3.metric(t("kpi_brands"), f"{df['make'].nunique():,}")

st.write("")
left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader(t("nav_price"))
    st.write(t("nav_price_desc"))
    st.page_link("pages/Price_Prediction.py", label=t("open_page"), icon="💰")

with right, st.container(border=True):
    st.subheader(t("nav_market"))
    st.write(t("nav_market_desc"))
    st.page_link("pages/Market_Clustering.py", label=t("open_page"), icon="📊")

st.write("")
with st.expander(t("limitations_title")):
    st.markdown(t("limitations", err=round(TYPICAL_ERROR * 100)))