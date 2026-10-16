import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    DEAL_THRESHOLD,
    REF_YEAR,
    TYPICAL_ERROR,
    label,
    load_data_with_tiers,
    load_model,
    sidebar,
    t,
)

st.set_page_config(page_title="Price Estimator", page_icon="💰", layout="wide")
sidebar()

model = load_model()
df = load_data_with_tiers()

st.title(t("price_title"))
st.write(t("price_intro"))
st.write("")

# ---------------- Dynamic Make/Model Selectors (Outside Form)
c1, c2 = st.columns(2)
with c1:
    makes = sorted(df["make"].dropna().unique().tolist())
    selected_make = st.selectbox(t("f_make"), makes)

with c2:
    filtered_models = sorted(
        df[df["make"] == selected_make]["model"].dropna().unique().tolist()
    )
    selected_model = st.selectbox(t("f_model"), filtered_models)

# ---------------- Car Attributes Form
with st.form("price_estimation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.slider(
            t("f_year"),
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=2018,
        )

        condition = st.radio(
            t("f_condition"),
            ["Used", "New"],
            format_func=lambda x: label(x),
            horizontal=True,
        )

    with col2:
        kilometers = st.number_input(
            t("f_km"),
            min_value=0,
            max_value=500000,
            value=80000,
            step=5000,
        )

        transmission = st.selectbox(
            t("f_transmission"),
            sorted(df["transmission"].dropna().unique().tolist()),
            format_func=lambda x: label(x),
        )

    with col3:
        fuel_type = st.selectbox(
            t("f_fuel"),
            sorted(df["fuel_type"].dropna().unique().tolist()),
            format_func=lambda x: label(x),
        )

        body_type = st.selectbox(
            t("f_body"),
            sorted(df["body_type"].dropna().unique().tolist()),
            format_func=lambda x: label(x),
        )

        colors = sorted(df["exterior_color"].dropna().unique().tolist())
        color = st.selectbox(
            t("f_color"),
            colors,
            format_func=lambda x: label(x),
        )

    submit = st.form_submit_button(t("submit"), use_container_width=True)

# ---------------- Output Prediction & Marketplace Matches
if submit:
    age = REF_YEAR - year

    input_data = pd.DataFrame(
        [
            {
                "make": selected_make,
                "model": selected_model,
                "age": age,
                "kilometers": kilometers,
                "transmission": transmission,
                "fuel_type": fuel_type,
                "body_type": body_type,
                "exterior_color": color,
                "condition": condition,
            }
        ]
    )

    est_price = float(model.predict(input_data)[0])
    lower = est_price * (1 - TYPICAL_ERROR)
    upper = est_price * (1 + TYPICAL_ERROR)

    st.write("")
    st.markdown("---")

    r1, r2 = st.columns([1, 2])
    with r1, st.container(border=True):
        st.metric(t("est_value"), f"{est_price:,.0f} {t('egp')}")
        st.caption(
            t(
                "est_caption",
                year=year,
                make=selected_make,
                model=selected_model,
                condition=label(condition),
                km=f"{kilometers:,}",
            )
        )

    with r2, st.container(border=True):
        st.subheader(t("est_range"))
        st.markdown(f"### **{lower:,.0f}** — **{upper:,.0f}** {t('egp')}")
        st.caption(t("err_note", pct=round(TYPICAL_ERROR * 100)))

    # ---------------- Similar Listings Table
    st.write("")
    st.subheader(t("similar_title"))

    matches = df[
        (df["make"] == selected_make)
        & (df["model"] == selected_model)
        & (df["year"].between(year - 2, year + 2))
    ].copy()

    is_fallback = False
    if matches.empty:
        matches = df[
            (df["make"] == selected_make)
            & (df["year"].between(year - 2, year + 2))
        ].copy()
        is_fallback = True

    if matches.empty:
        st.info(t("no_matches"))
    else:
        if is_fallback:
            st.info(t("fallback_info"))
        else:
            same_model_count = len(matches[matches["model"] == selected_model])
            st.caption(
                t("same_model_info", n=same_model_count, total=len(matches))
            )

        def classify_deal(p):
            diff = (p - est_price) / est_price
            pct = abs(round(diff * 100))
            if diff < -DEAL_THRESHOLD:
                return t("deal_below", pct=pct)
            elif diff > DEAL_THRESHOLD:
                return t("deal_above", pct=pct)
            return t("deal_fair")

        matches["deal"] = matches["price"].apply(classify_deal)

        # Detect the exact URL column from dataset
        url_col = None
        for col_name in ["url", "link", "listing_url", "source_url"]:
            if col_name in matches.columns:
                url_col = col_name
                break

        if url_col:
            display_cols = ["model", "year", "kilometers", "condition", "price", "deal", url_col]
        else:
            display_cols = ["model", "year", "kilometers", "condition", "price", "deal"]

        display_df = matches[display_cols].head(10).copy()
        display_df["condition"] = display_df["condition"].apply(label)

        column_configuration = {
            "model": t("f_model"),
            "year": t("f_year"),
            "kilometers": st.column_config.NumberColumn(
                t("km_label"), format="%d km"
            ),
            "condition": t("cond_label"),
            "price": st.column_config.NumberColumn(
                t("price_label"), format="%d EGP"
            ),
            "deal": "Status",
        }

        if url_col:
            column_configuration[url_col] = st.column_config.LinkColumn(
                t("view_listing"),
                display_text=t("view_listing"),
            )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config=column_configuration,
        )