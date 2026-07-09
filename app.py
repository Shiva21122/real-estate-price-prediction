"""
Real Estate Price Predictor - interactive Streamlit app.

The price estimate updates in real time as you describe the property in the
sidebar. Tabs cover the live estimate with sensitivity analysis and
comparable sales, an explorable market view, and model insights
(feature importances).
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠",
                   layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["year_sold", "property_tax", "insurance", "beds", "baths", "sqft",
            "year_built", "lot_size", "basement", "popular", "recession",
            "property_age", "property_type_Condo"]


# ─── Cached loaders ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(HERE, "models", "real_estate_model.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(HERE, "data", "final.csv"))


model = load_model()
df = load_data()


# ─── Sidebar: describe the property (live - no button) ───────────────────
st.sidebar.header("🏠 Describe the Property")
st.sidebar.caption("The estimate updates instantly as you change values.")

prop_type = st.sidebar.radio("Property type", ["House", "Condo"], horizontal=True)
sqft = st.sidebar.slider("Living area (sqft)", 500, 8000, 1900, 50)
beds = st.sidebar.slider("Bedrooms", 1, 5, 3)
baths = st.sidebar.slider("Bathrooms", 1, 6, 2)
year_built = st.sidebar.slider("Year built", 1880, 2014, 1990)
year_sold = st.sidebar.slider("Year sold", 1993, 2016, 2010)
lot_size = st.sidebar.number_input("Lot size (sqft)", 0, 450000, 5900, 100)
property_tax = st.sidebar.number_input("Monthly property tax ($)", 0, 5000, 420, 10)
insurance = st.sidebar.number_input("Monthly insurance ($)", 0, 1500, 125, 5)
basement = st.sidebar.toggle("Has basement", value=True)
popular = st.sidebar.toggle("Popular neighbourhood", value=False)
recession = st.sidebar.toggle("Sold during recession", value=False)

property_age = max(year_sold - year_built, 0)
is_condo = 1 if prop_type == "Condo" else 0


def build_row(**overrides):
    row = {
        "year_sold": year_sold, "property_tax": property_tax,
        "insurance": insurance, "beds": beds, "baths": baths, "sqft": sqft,
        "year_built": year_built, "lot_size": lot_size,
        "basement": int(basement), "popular": int(popular),
        "recession": int(recession), "property_age": property_age,
        "property_type_Condo": is_condo,
    }
    row.update(overrides)
    return pd.DataFrame([row])[FEATURES]


price = float(model.predict(build_row())[0])

# ─── Header + live headline metrics ──────────────────────────────────────
st.title("🏠 Real Estate Price Predictor")

similar = df[(df["property_type_Condo"] == is_condo)
             & df["beds"].between(beds - 1, beds + 1)
             & df["sqft"].between(sqft * 0.8, sqft * 1.2)]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated Price", f"${price:,.0f}")
m2.metric("Price per sqft", f"${price / sqft:,.0f}")
m3.metric("Market median (all)", f"${df['price'].median():,.0f}",
          delta=f"{(price / df['price'].median() - 1):+.0%} vs your estimate")
m4.metric("Similar homes in data", f"{len(similar)}",
          delta=(f"median ${similar['price'].median():,.0f}"
                 if len(similar) else "none found"))

tab_estimate, tab_market, tab_model = st.tabs(
    ["💰 Price Estimate", "📊 Explore the Market", "🧠 Model Insights"])


# ─── Tab 1: estimate, sensitivity, comparables ───────────────────────────
with tab_estimate:
    left, right = st.columns(2)

    with left:
        st.subheader("What moves the price? (live sensitivity)")
        st.caption("Predicted price as ONE feature varies, others fixed at "
                   "your current inputs.")

        sqft_range = np.arange(600, 5001, 200)
        sqft_prices = [float(model.predict(build_row(sqft=int(v)))[0])
                       for v in sqft_range]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sqft_range, sqft_prices, lw=2.5, color="#2c7fb8")
        ax.scatter([sqft], [price], s=250, marker="*", c="#e6550d",
                   zorder=5, label="Your property")
        ax.set_xlabel("Living area (sqft)")
        ax.set_ylabel("Predicted price ($)")
        ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}K")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        b1, b2 = st.columns(2)
        with b1:
            bed_prices = pd.Series(
                {b: float(model.predict(build_row(beds=b))[0])
                 for b in range(1, 6)}, name="price")
            st.caption("Price by bedroom count")
            st.bar_chart(bed_prices, height=200)
        with b2:
            bath_prices = pd.Series(
                {b: float(model.predict(build_row(baths=b))[0])
                 for b in range(1, 7)}, name="price")
            st.caption("Price by bathroom count")
            st.bar_chart(bath_prices, height=200)

        flip_type = float(model.predict(
            build_row(property_type_Condo=1 - is_condo))[0])
        other = "Condo" if prop_type == "House" else "House"
        st.info(f"As a **{other}** instead of a {prop_type}, this property "
                f"would be estimated at **${flip_type:,.0f}** "
                f"({(flip_type / price - 1):+.1%}).")

    with right:
        st.subheader("Comparable sales in the dataset")
        st.caption(f"{prop_type}s with {beds}±1 beds and sqft within ±20% "
                   f"of yours.")
        if len(similar):
            comps = similar.copy()
            comps["vs your estimate"] = comps["price"] - price
            show_cols = ["price", "sqft", "beds", "baths", "year_built",
                         "year_sold", "vs your estimate"]
            table = (comps[show_cols]
                     .sort_values("vs your estimate", key=abs).head(12).copy())
            table["price"] = table["price"].map("${:,.0f}".format)
            table["vs your estimate"] = table["vs your estimate"].map(
                "${:+,.0f}".format)
            st.dataframe(table, width="stretch", height=300)

            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(similar["price"], bins=20, color="#74c476", ax=ax)
            ax.axvline(price, color="#e6550d", lw=3, ls="--")
            ax.text(price, ax.get_ylim()[1] * 0.9, "  Your estimate",
                    color="#e6550d", fontweight="bold")
            ax.set_xlabel("Sale price ($)")
            ax.xaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}K")
            st.pyplot(fig)
            plt.close(fig)
            pct = (similar["price"] < price).mean()
            st.caption(f"Your estimate is higher than **{pct:.0%}** of "
                       f"comparable sales.")
        else:
            st.warning("No comparable properties found - try adjusting "
                       "sqft or bedrooms.")


# ─── Tab 2: market explorer with live filters ────────────────────────────
with tab_market:
    f1, f2, f3 = st.columns(3)
    type_filter = f1.radio("Type", ["All", "Houses", "Condos"], horizontal=True)
    year_range = f2.slider("Year sold", 1993, 2016, (1993, 2016))
    price_range = f3.slider("Price range ($K)", 200, 800, (200, 800), 25)

    view = df.copy()
    if type_filter == "Houses":
        view = view[view["property_type_Condo"] == 0]
    elif type_filter == "Condos":
        view = view[view["property_type_Condo"] == 1]
    view = view[view["year_sold"].between(*year_range)
                & view["price"].between(price_range[0] * 1000,
                                        price_range[1] * 1000)]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Properties in view", len(view))
    s2.metric("Median price", f"${view['price'].median():,.0f}" if len(view) else "-")
    s3.metric("Median $/sqft",
              f"${(view['price'] / view['sqft']).median():,.0f}" if len(view) else "-")
    s4.metric("Median sqft", f"{view['sqft'].median():,.0f}" if len(view) else "-")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Size vs price - you are the star ⭐")
        fig, ax = plt.subplots(figsize=(7, 5))
        sample = view.sample(min(len(view), 1500), random_state=1) if len(view) else view
        sns.scatterplot(data=sample, x="sqft", y="price",
                        hue="property_type_Condo",
                        palette={0: "#3182bd", 1: "#e6550d"},
                        alpha=0.4, ax=ax)
        ax.scatter([sqft], [price], marker="*", s=600, c="#31a354",
                   edgecolors="black", zorder=5, label="Your property")
        handles, labels = ax.get_legend_handles_labels()
        labels = ["House" if l == "0" else "Condo" if l == "1" else l
                  for l in labels]
        ax.legend(handles, labels)
        ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}K")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Market trend over time")
        if len(view):
            trend = view.groupby("year_sold")["price"].median()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(trend.index, trend.values, marker="o", lw=2.5,
                    color="#756bb1")
            rec_years = view[view["recession"] == 1]["year_sold"].unique()
            for y in rec_years:
                ax.axvspan(y - 0.5, y + 0.5, color="red", alpha=0.08)
            ax.set_xlabel("Year sold")
            ax.set_ylabel("Median price ($)")
            ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}K")
            ax.grid(alpha=0.3)
            ax.set_title("Median sale price by year (recession years shaded)")
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("🔎 Browse the filtered market data"):
        st.dataframe(view, width="stretch", height=300)
        st.download_button("Download filtered data as CSV",
                           view.to_csv(index=False), "market_filtered.csv",
                           "text/csv")


# ─── Tab 3: model insights ───────────────────────────────────────────────
with tab_model:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("What the model cares about")
        importances = (pd.Series(model.feature_importances_, index=FEATURES)
                       .sort_values())
        fig, ax = plt.subplots(figsize=(7, 5))
        importances.plot.barh(ax=ax, color="#2c7fb8")
        ax.set_xlabel("Feature importance")
        ax.set_title("Random Forest feature importances")
        st.pyplot(fig)
        plt.close(fig)
        top = importances.sort_values(ascending=False)
        st.caption(f"Top drivers: **{top.index[0]}** ({top.iloc[0]:.0%}), "
                   f"**{top.index[1]}** ({top.iloc[1]:.0%}), "
                   f"**{top.index[2]}** ({top.iloc[2]:.0%}).")

    with c2:
        st.subheader("Model card")
        st.markdown("""
| | |
|---|---|
| Algorithm | Random Forest (200 trees, MAE criterion) |
| Test MAE | ~$45,000 |
| Baseline (Linear Regression) MAE | ~$84,000 |
| Training data | 3,000+ sales, 1993-2016 |
| Features | 13 (size, rooms, age, taxes, market timing, type) |
""")
        tree_path = os.path.join(HERE, "assets", "tree.png")
        if os.path.exists(tree_path):
            with st.expander("🌳 View a single decision tree (illustrative)"):
                st.image(tree_path, width="stretch")
        st.caption("Retrain with `python train_model.py`.")

st.divider()
st.caption("Educational demo - not intended for real-world pricing decisions.")
