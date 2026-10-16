"""Shared helpers for the Egyptian used-car app: settings, translations,
data and model loading, clustering, and page styling."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Egypt_Used_Cars_clean.csv"
MODEL_PATH = ROOT / "xgboost_car_price_model.pkl"

REF_YEAR = 2027        # must equal the REF_YEAR used in the notebook (age = REF_YEAR - year)
TYPICAL_ERROR = 0.11   # median % error of the deployed model on the test set (about 10.6%)
DEAL_THRESHOLD = 0.10  # listings within +-10% of the estimate count as "close"
TIER_COLORS = ["#F59E0B", "#0EA5E9", "#8B5CF6"]  # tier 0 = Budget ... tier 2 = Luxury

MODEL_FEATURES_CAT = [
    "make", "model", "transmission", "fuel_type",
    "body_type", "exterior_color", "condition",
]

# Test-set results copied from notebook
MODEL_RESULTS = [
    ("Linear Regression", 22.2, 12.8, 0.655),
    ("Ridge", 22.2, 12.8, 0.655),
    ("Lasso", 27.2, 18.5, 0.553),
    ("Random Forest (tuned)", 17.0, 8.9, 0.748),
    ("XGBoost (tuned), deployed", 18.5, 10.6, 0.727),
]

# ------------------------------------------------------------ translations
T = {
    "en": {
        "app_title": "🚗 Egyptian Used Car Market",
        "app_subtitle": "Estimate a fair price for a used car in Egypt and explore how the market splits into groups, based on thousands of real listings.",
        "kpi_listings": "Listings analysed",
        "kpi_median_price": "Median price",
        "kpi_brands": "Brands",
        "egp": "EGP",
        "open_page": "Open page",
        "nav_price": "Price Estimator",
        "nav_price_desc": "Enter your car's details to get an estimated fair price, then compare similar listings against it.",
        "nav_market": "Market Segments",
        "nav_market_desc": "Explore the three market groups found by clustering, filtered by budget, brand and condition.",
        "model_perf": "How accurate is the model?",
        "model_perf_note": "Results on test listings the model never saw. The deployed model is XGBoost.",
        "col_model": "Model",
        "col_mape": "Mean % error",
        "col_medape": "Median % error",
        "col_r2": "R²",
        "limitations_title": "Limitations",
        "limitations": "- Prices are **asking prices** from listings, not final sale prices.\n- The data is a snapshot from several sources.\n- Rare models have few listings, so estimates are less reliable.\n- The typical error is about ±{err}%.",
        "price_title": "💰 Price Estimator",
        "price_intro": "Enter your car's details to get an estimated fair price and similar listings from the dataset.",
        "f_make": "Brand",
        "f_model": "Model",
        "f_year": "Manufacturing year",
        "f_km": "Kilometers driven",
        "f_condition": "Condition",
        "f_transmission": "Transmission",
        "f_fuel": "Fuel type",
        "f_body": "Body style",
        "f_color": "Exterior color",
        "unknown": "Not specified",
        "submit": "Estimate price",
        "est_value": "Estimated fair price",
        "est_range": "Likely range",
        "est_caption": "For a {year} {make} {model} ({condition}) with {km} km.",
        "err_note": "This model's typical error is about ±{pct}%. Use the result as a guide, not a quote.",
        "similar_title": "Similar listings",
        "same_model_info": "{n} of the {total} listings below are the same model.",
        "fallback_info": "No close listings for this exact model, so these are the closest cars from the same brand.",
        "no_matches": "No listings found for this brand in the dataset.",
        "deal_below": "🟢 {pct}% below the estimate",
        "deal_fair": "🟡 Close to the estimate",
        "deal_above": "🔴 {pct}% above the estimate",
        "view_listing": "View listing",
        "chart_title": "Prices of {make} {model} listings",
        "your_estimate": "Your estimate",
        "price_label": "Price",
        "km_label": "Mileage",
        "cond_label": "Condition",
        "km_unit": "km",
        "market_title": "📊 Market Segments",
        "market_intro": "Clustering groups similar cars together. Each group below is described with the real data in your selection.",
        "filters": "Filters",
        "budget": "Budget (EGP)",
        "brands": "Brands",
        "brands_help": "Leave empty to include all brands",
        "cond_filter": "Condition",
        "cond_all": "All",
        "showing": "Showing **{n:,}** listings",
        "tier_0_name": "Economy / Budget Tier",
        "tier_0_desc": "Affordable entry-level vehicles and daily drivers across the market.",
        "tier_1_name": "Mid-Range Segment",
        "tier_1_desc": "Popular mid-tier sedans and crossover vehicles offering strong overall value.",
        "tier_2_name": "Luxury & High-End Tier",
        "tier_2_desc": "Premium brands, late-model SUVs, and upper-tier market listings.",
        "t_count": "Listings",
        "t_price": "Median price",
        "t_age": "Median age",
        "t_km": "Median mileage",
        "t_brands": "Top brands",
        "years": "years",
        "tab_scatter": "Price vs. year",
        "tab_profile": "Group profiles",
        "no_results": "No listings match these filters. Try widening the budget.",
        "x_year": "Year",
        "y_price": "Price (EGP, log scale)",
        "group": "Group",
        "sample_note": "Showing a random sample of {n:,} points for speed.",
        "median_price_by_group": "Median price by group",
    },
    "ar": {
        "app_title": "🚗 سوق السيارات المستعملة في مصر",
        "app_subtitle": "قدّر السعر العادل لسيارة مستعملة في مصر، واستكشف كيف ينقسم السوق إلى مجموعات، اعتماداً على آلاف الإعلانات الحقيقية.",
        "kpi_listings": "عدد الإعلانات",
        "kpi_median_price": "السعر الوسيط",
        "kpi_brands": "عدد الماركات",
        "egp": "جنيه",
        "open_page": "افتح الصفحة",
        "nav_price": "تقدير السعر",
        "nav_price_desc": "أدخل بيانات سيارتك لتحصل على سعر عادل تقديري، ثم قارن به إعلانات مشابهة.",
        "nav_market": "فئات السوق",
        "nav_market_desc": "استكشف المجموعات الثلاث الناتجة عن التجميع، وصفّها حسب الميزانية والماركة والحالة.",
        "model_perf": "ما مدى دقة النموذج؟",
        "model_perf_note": "نتائج على إعلانات اختبار لم يرها النموذج. النموذج المستخدم هو XGBoost.",
        "col_model": "النموذج",
        "col_mape": "متوسط الخطأ %",
        "col_medape": "الخطأ الوسيط %",
        "col_r2": "R²",
        "limitations_title": "حدود النموذج",
        "limitations": "- الأسعار هي **أسعار الطلب** في الإعلانات وليست أسعار البيع النهائية.\n- البيانات لقطة من عدة مصادر.\n- الموديلات النادرة لها إعلانات قليلة، فتقديراتها أقل دقة.\n- الخطأ المعتاد حوالي ±{err}%.",
        "price_title": "💰 تقدير سعر السيارة",
        "price_intro": "أدخل بيانات سيارتك لتحصل على سعر عادل تقديري وإعلانات مشابهة من قاعدة البيانات.",
        "f_make": "الماركة",
        "f_model": "الموديل",
        "f_year": "سنة الصنع",
        "f_km": "الكيلومترات",
        "f_condition": "الحالة",
        "f_transmission": "ناقل الحركة",
        "f_fuel": "نوع الوقود",
        "f_body": "نمط الهيكل",
        "f_color": "اللون الخارجي",
        "unknown": "غير محدد",
        "submit": "احسب السعر",
        "est_value": "السعر العادل التقديري",
        "est_range": "النطاق المتوقع",
        "est_caption": "لسيارة {make} {model} موديل {year} ({condition}) قطعت {km} كم.",
        "err_note": "الخطأ المعتاد للنموذج حوالي ±{pct}%. اعتبر النتيجة مؤشراً وليست عرض سعر.",
        "similar_title": "إعلانات مشابهة",
        "same_model_info": "{n} من أصل {total} إعلانات أدناه من نفس الموديل.",
        "fallback_info": "لا توجد إعلانات قريبة لنفس الموديل، لذلك نعرض أقرب سيارات من نفس الماركة.",
        "no_matches": "لا توجد إعلانات لهذه الماركة في قاعدة البيانات.",
        "deal_below": "🟢 أقل من التقدير بنسبة {pct}%",
        "deal_fair": "🟡 قريب من التقدير",
        "deal_above": "🔴 أعلى من التقدير بنسبة {pct}%",
        "view_listing": "عرض الإعلان",
        "chart_title": "أسعار إعلانات {make} {model}",
        "your_estimate": "تقديرك",
        "price_label": "السعر",
        "km_label": "الكيلومترات",
        "cond_label": "الحالة",
        "km_unit": "كم",
        "market_title": "📊 فئات السوق",
        "market_intro": "التجميع يضع السيارات المتشابهة في مجموعات. كل مجموعة موصوفة أدناه ببيانات اختيارك الفعلية.",
        "filters": "التصفية",
        "budget": "الميزانية (جنيه)",
        "brands": "الماركات",
        "brands_help": "اتركها فارغة لعرض كل الماركات",
        "cond_filter": "الحالة",
        "cond_all": "الكل",
        "showing": "عرض **{n:,}** إعلان",
        "tier_0_name": "الفئة الاقتصادية",
        "tier_0_desc": "سيارات منخفضة التكلفة تناسب الاستخدام اليومي والميزانيات المحدودة.",
        "tier_1_name": "الفئة المتوسطة",
        "tier_1_desc": "سيارات سيدان وكروس أوفر واسعة الانتشار تقدم قيمة متوازنة مقابل السعر.",
        "tier_2_name": "الفئة الفاخرة والعالية",
        "tier_2_desc": "الماركات الفاخرة والسيارات الحديثة التي تمثل الشريحة السعرية الأعلى.",
        "t_count": "عدد الإعلانات",
        "t_price": "السعر الوسيط",
        "t_age": "العمر الوسيط",
        "t_km": "الكيلومترات الوسيطة",
        "t_brands": "أشهر الماركات",
        "years": "سنة",
        "tab_scatter": "السعر مقابل السنة",
        "tab_profile": "ملف كل مجموعة",
        "no_results": "لا توجد إعلانات مطابقة. جرّب توسيع الميزانية.",
        "x_year": "السنة",
        "y_price": "السعر (جنيه، مقياس لوغاريتمي)",
        "group": "المجموعة",
        "sample_note": "نعرض عينة عشوائية من {n:,} نقطة لتسريع العرض.",
        "median_price_by_group": "السعر الوسيط لكل مجموعة",
    },
}

LABELS_AR = {
    "Used": "مستعملة", "New": "جديدة",
    "Automatic": "أوتوماتيك", "Manual": "مانيوال",
    "Benzine": "بنزين", "Diesel": "ديزل", "Natural Gas": "غاز طبيعي",
    "Hybrid": "هايبرد", "Electric": "كهرباء",
    "Sedan": "سيدان", "SUV": "دفع رباعي (SUV)", "Hatchback": "هاتشباك",
    "Sports / Coupe": "كوبيه / رياضية", "Van / Bus": "فان / ميني باص",
    "Pickup": "بيك أب", "Convertible": "كابريوليه", "MPV": "عائلية (MPV)",
    "Estate": "ستيشن", "4X4": "دفع رباعي (4X4)",
    "Black": "أسود", "Silver": "فضي", "White": "أبيض", "Red": "أحمر",
    "Blue- Navy Blue": "أزرق / كحلي", "Green": "أخضر", "Beige": "بيج",
    "Burgundy": "بورجوندي", "Yellow": "أصفر", "Orange": "برتقالي",
    "Gray": "رمادي", "Other Color": "لون آخر", "Brown": "بني", "Gold": "ذهبي",
    "Dark Grey": "رمادي غامق", "Bronze": "برونزي", "Maroon": "ماروني",
    "Baby Blue": "سماوي", "Turquoise": "تركواز", "Champagne": "شامبانيا",
    "Purple": "بنفسجي",
}


def lang() -> str:
    return st.session_state.get("lang", "en")


def t(key: str, **kwargs) -> str:
    text = T[lang()][key]
    return text.format(**kwargs) if kwargs else text


def label(value) -> str:
    if not isinstance(value, str):
        return "-"
    return LABELS_AR.get(value, value) if lang() == "ar" else value


# ----------------------------------------------------------------- styling
_BASE_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stMetricValue"] { font-size: 2.1rem; }
</style>
"""

_RTL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

/* Apply Cairo font to standard markdown text only */
html, body, p, [data-testid="stMarkdownContainer"] p { 
    font-family: 'Cairo', sans-serif; 
}

/* RTL Text Alignment for normal markdown paragraphs and labels */
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"],
[data-testid="stMetric"] {
    direction: rtl; 
    text-align: right;
}

/* --- FIX MATERIAL ICON LIGATURE 'keyboard_ar' OVERLAP --- */
/* Exclude Material Icons from receiving font-family override */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
details summary span {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    direction: ltr !important;
}

/* Expander layout styling in RTL mode */
[data-testid="stExpander"] details summary {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    direction: rtl;
    text-align: right;
}
</style>
"""


def sidebar() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    choice = st.sidebar.radio(
        "Language / اللغة",
        ["English", "العربية"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
    )
    st.session_state.lang = "en" if choice == "English" else "ar"
    st.markdown(_BASE_CSS + (_RTL_CSS if lang() == "ar" else ""), unsafe_allow_html=True)


# ------------------------------------------------------------ data / model
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "year" not in df.columns:
        df["year"] = REF_YEAR - df["age"]
    if "age" not in df.columns:
        df["age"] = REF_YEAR - df["year"]
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def _cluster_ids(df: pd.DataFrame) -> np.ndarray:
    """KMeans (k=3) including log-transformed PRICE to build real market segments
    (Budget vs. Mid-Range vs. Luxury/High-End). Ranked by median price: 0 = lowest ... 2 = highest."""
    df_copy = df.copy()
    df_copy["log_price"] = np.log1p(df_copy["price"])

    prep = ColumnTransformer([
        ("price_num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler())]), ["log_price"]),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("pt", PowerTransformer())]), ["age"]),
        ("km", Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())]), ["kilometers"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                min_frequency=20, sparse_output=False))]),
         MODEL_FEATURES_CAT),
    ])

    X = prep.fit_transform(df_copy[["log_price", "age", "kilometers"] + MODEL_FEATURES_CAT])
    ids = KMeans(n_clusters=3, random_state=11, n_init=10).fit_predict(X)

    order = (pd.Series(df_copy["price"].to_numpy()).groupby(ids).median()
             .sort_values(ascending=True).index.tolist())
    rank = {cid: r for r, cid in enumerate(order)}
    return np.array([rank[i] for i in ids])


@st.cache_data(show_spinner=False)
def load_data_with_tiers() -> pd.DataFrame:
    df = load_data().copy()
    df["tier"] = _cluster_ids(df)
    return df