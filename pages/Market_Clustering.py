import pandas as pd
import plotly.express as px
import streamlit as st

from utils import TIER_COLORS, label, load_data_with_tiers, sidebar, t

st.set_page_config(page_title="Market Segments", page_icon="📊", layout="wide")
sidebar()

df = load_data_with_tiers()
tier_names = {i: t(f"tier_{i}_name") for i in range(3)}

# Known premium/luxury brands in the Egyptian market
LUXURY_BRANDS = {
    "Mercedes-Benz", "BMW", "Audi", "Porsche", "Land Rover", 
    "Jaguar", "Lexus", "Volvo", "Alfa Romeo", "Jeep", "Mini"
}

# ------------------------------------------------------------ Sidebar Filters
st.sidebar.header(t("filters"))
low = int(df["price"].min())
high = int(df["price"].quantile(0.98))
budget = st.sidebar.slider(
    t("budget"),
    min_value=low,
    max_value=high,
    value=(max(low, 150_000), min(high, 3_500_000)),
    step=25_000,
)
brands = st.sidebar.multiselect(
    t("brands"), sorted(df["make"].dropna().unique().tolist()), help=t("brands_help")
)
condition = st.sidebar.radio(
    t("cond_filter"),
    [None, "Used", "New"],
    format_func=lambda v: t("cond_all") if v is None else label(v),
)

view = df[df["price"].between(*budget)]
if brands:
    view = view[view["make"].isin(brands)]
if condition:
    view = view[view["condition"] == condition]

# ---------------------------------------------------------------- Content
st.title(t("market_title"))
st.write(t("market_intro"))
st.markdown(t("showing", n=len(view)))

if view.empty:
    st.warning(t("no_results"))
    st.stop()

cards = st.columns(3)
for i, col in enumerate(cards):
    seg = view[view["tier"] == i]
    with col, st.container(border=True):
        st.markdown(
            f"<div style='height:6px;border-radius:3px;background:{TIER_COLORS[i]}'></div>",
            unsafe_allow_html=True,
        )
        st.subheader(tier_names[i])
        st.caption(t(f"tier_{i}_desc"))
        
        if seg.empty:
            st.write("-")
            continue
            
        st.metric(t("t_price"), f"{seg['price'].median():,.0f} {t('egp')}")
        st.write(f"{t('t_count')}: **{len(seg):,}**")
        st.write(f"{t('t_age')}: **{seg['age'].median():.0f} {t('years')}**")
        st.write(f"{t('t_km')}: **{seg['kilometers'].median():,.0f} {t('km_unit')}**")
        
        # Filter top brands based on tier true identity
        if i == 2 and not brands:
            # For Tier 2 (Luxury), prioritize actual luxury marques present in the tier
            lux_in_seg = seg[seg["make"].isin(LUXURY_BRANDS)]
            if not lux_in_seg.empty:
                top_b = lux_in_seg["make"].value_counts().head(3).index.tolist()
            else:
                top_b = seg["make"].value_counts().head(3).index.tolist()
        else:
            top_b = seg["make"].value_counts().head(3).index.tolist()

        st.write(f"{t('t_brands')}: **{', '.join(top_b)}**")

tab_scatter, tab_profile = st.tabs([t("tab_scatter"), t("tab_profile")])

with tab_scatter:
    plot = view.assign(group=view["tier"].map(tier_names))
    if len(plot) > 8_000:
        plot = plot.sample(8_000, random_state=11)
        st.caption(t("sample_note", n=len(plot)))
    fig = px.scatter(
        plot,
        x="year",
        y="price",
        color="group",
        log_y=True,
        opacity=0.6,
        color_discrete_map={tier_names[i]: TIER_COLORS[i] for i in range(3)},
        category_orders={"group": [tier_names[i] for i in range(3)]},
        hover_data={
            "make": True,
            "model": True,
            "kilometers": ":,.0f",
            "price": ":,.0f",
            "group": False,
        },
        labels={"year": t("x_year"), "price": t("y_price"), "group": t("group")},
        height=520,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_profile:
    profile = (
        view.groupby("tier")
        .agg(
            count=("price", "size"),
            price=("price", "median"),
            age=("age", "median"),
            km=("kilometers", "median"),
        )
        .reindex(range(3))
        .rename(index=tier_names)
        .reset_index()
        .rename(
            columns={
                "tier": t("group"),
                "count": t("t_count"),
                "price": t("t_price"),
                "age": t("t_age"),
                "km": t("t_km"),
            }
        )
    )
    bar = px.bar(
        profile,
        x=t("group"),
        y=t("t_price"),
        color=t("group"),
        color_discrete_map={tier_names[i]: TIER_COLORS[i] for i in range(3)},
        title=t("median_price_by_group"),
        height=360,
    )
    bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(profile, hide_index=True, use_container_width=True)