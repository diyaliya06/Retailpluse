"""RetailPulse Streamlit dashboard.

Run from the RetailPulse folder:
    python -m streamlit run src/app.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import ART_DIR  # noqa: E402

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")


@st.cache_data
def load_csv(name, date_cols=()):
    path = os.path.join(ART_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data
def load_json(name):
    path = os.path.join(ART_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def missing(what):
    st.info(f"{what} not found. Run `python -m src.train` first.")


st.title("📊 RetailPulse")
st.subheader("AI-Powered Customer Analytics & Demand Forecasting Platform")
st.caption("Customer segmentation · churn prediction · demand forecasting · "
           "inventory optimisation · model monitoring")

tx = load_csv("cleaned_transactions.csv", ("InvoiceDate",))
if tx is None:
    st.warning(
        "Cleaned dataset not found. Put your raw file in the `data` folder, "
        "then run: `python -m src.train`"
    )
    st.stop()

tab_overview, tab_seg, tab_churn, tab_fc, tab_inv, tab_mon = st.tabs([
    "📈 Overview", "👥 Segments", "⚠️ Churn",
    "🔮 Forecast", "📦 Inventory", "🛡️ Monitoring",
])

# ------------------------------------------------------------------ Overview
with tab_overview:
    countries = sorted(tx["Country"].dropna().unique())
    chosen = st.multiselect("Country filter (empty = all)", countries)
    view = tx[tx["Country"].isin(chosen)] if chosen else tx

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"£{view['Revenue'].sum():,.0f}")
    c2.metric("Transactions", f"{view['Invoice'].nunique():,}")
    c3.metric("Customers", f"{view['CustomerID'].nunique():,}")
    c4.metric("Products", f"{view['StockCode'].nunique():,}")

    monthly = (view.assign(Month=view["InvoiceDate"].dt.to_period("M").astype(str))
               .groupby("Month")["Revenue"].sum().reset_index())
    st.plotly_chart(px.line(monthly, x="Month", y="Revenue", markers=True,
                            title="Monthly Revenue"))

    top = (view.groupby("Description")["Revenue"].sum()
           .nlargest(10).reset_index().sort_values("Revenue"))
    st.plotly_chart(px.bar(top, x="Revenue", y="Description", orientation="h",
                           title="Top 10 Products by Revenue"))

# ------------------------------------------------------------------ Segments
with tab_seg:
    rfm = load_csv("rfm_customers.csv")
    if rfm is None:
        missing("RFM file")
    else:
        left, right = st.columns(2)
        counts = rfm["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Customers"]
        left.plotly_chart(px.pie(counts, names="Segment", values="Customers",
                                 title="Customer Segments"))
        right.plotly_chart(px.scatter(
            rfm, x="Recency", y="Monetary", color="Segment",
            hover_data=["CustomerID", "Frequency"],
            title="Recency vs Monetary"))
        summary = (rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
                   .mean().round(1).reset_index())
        st.markdown("**Average RFM per segment**")
        st.dataframe(summary)
        st.markdown("**Customers**")
        st.dataframe(rfm)

# --------------------------------------------------------------------- Churn
with tab_churn:
    churn = load_csv("churn_predictions.csv")
    if churn is None:
        missing("Churn predictions")
    else:
        churn["Risk"] = pd.cut(
            churn["ChurnProbability"], [-0.01, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Average churn probability",
                  f"{churn['ChurnProbability'].mean():.1%}")
        m2.metric("High-risk customers", int((churn["Risk"] == "High").sum()))
        metrics = load_json("churn_metrics.json")
        if metrics and metrics.get("accuracy") is not None:
            m3.metric("Model accuracy", f"{metrics['accuracy']:.1%}")

        left, right = st.columns(2)
        risk = churn["Risk"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
        risk.columns = ["Risk", "Customers"]
        left.plotly_chart(px.bar(risk, x="Risk", y="Customers", color="Risk",
                                 title="Churn Risk Distribution",
                                 color_discrete_map={"Low": "#2a9d8f",
                                                     "Medium": "#e9c46a",
                                                     "High": "#e76f51"}))
        imp = load_csv("churn_feature_importance.csv")
        if imp is not None:
            right.plotly_chart(px.bar(imp.sort_values("Importance"),
                                      x="Importance", y="Feature",
                                      orientation="h",
                                      title="Churn Feature Importance"))
        else:
            right.info("Feature importance file is empty. Re-run the pipeline.")

        st.markdown("**Top 20 customers most likely to churn**")
        st.dataframe(churn.sort_values("ChurnProbability", ascending=False).head(20))

# ------------------------------------------------------------------ Forecast
with tab_fc:
    fc = load_csv("product_forecast.csv", ("ds",))
    if fc is None:
        missing("Forecast file")
    else:
        fc["StockCode"] = fc["StockCode"].astype(str)
        product = st.selectbox("Select product", sorted(fc["StockCode"].unique()))
        f = fc[fc["StockCode"] == product]

        hist = (tx[tx["StockCode"].astype(str) == product]
                .set_index("InvoiceDate")["Quantity"].resample("D").sum()
                .reset_index().tail(90))

        fig = go.Figure()
        fig.add_scatter(x=hist["InvoiceDate"], y=hist["Quantity"],
                        name="Actual (last 90 days)", mode="lines")
        fig.add_scatter(x=f["ds"], y=f["yhat_upper"], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=f["ds"], y=f["yhat_lower"], mode="lines",
                        line=dict(width=0), fill="tonexty",
                        name="Uncertainty range")
        fig.add_scatter(x=f["ds"], y=f["yhat"], name="Forecast", mode="lines")
        fig.update_layout(title=f"Demand forecast - {product}",
                          yaxis_title="Units per day")
        st.plotly_chart(fig)

        if "Model" in f.columns:
            st.caption(f"Selected model: **{f['Model'].iloc[0]}**")
        fm = load_csv("forecast_metrics.csv")
        if fm is not None:
            st.markdown("**Model comparison (hold-out period)**")
            st.dataframe(fm[fm["StockCode"].astype(str) == product]
                         .drop(columns="StockCode"))

# ----------------------------------------------------------------- Inventory
with tab_inv:
    inv = load_csv("inventory_recommendations.csv")
    if inv is None:
        missing("Inventory recommendations")
    else:
        priorities = st.multiselect("Priority", ["High", "Medium", "Low"],
                                    default=["High", "Medium", "Low"])
        view = inv[inv["Priority"].isin(priorities)]
        k1, k2, k3 = st.columns(3)
        k1.metric("Products", len(view))
        k2.metric("Total suggested reorder", f"{int(view['SuggestedReorderQty'].sum()):,}")
        k3.metric("High priority", int((inv["Priority"] == "High").sum()))
        st.plotly_chart(px.bar(
            view, x="StockCode", y="SuggestedReorderQty", color="Priority",
            title="Suggested Reorder Quantity",
            color_discrete_map={"High": "#e76f51", "Medium": "#e9c46a",
                                "Low": "#2a9d8f"}))
        st.dataframe(view)
        st.download_button("Download recommendations (CSV)",
                           view.to_csv(index=False), "inventory_recommendations.csv")

# ---------------------------------------------------------------- Monitoring
with tab_mon:
    drift = load_json("drift_report.json")
    if drift is None:
        missing("Drift report")
    else:
        table = pd.DataFrame(drift).T.reset_index().rename(columns={"index": "Feature"})
        for col in ("before_mean", "after_mean", "relative_change", "ks_pvalue"):
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce")
        table["drift"] = table["drift"].astype(bool)
        table["relative_change"] = (table["relative_change"] * 100).round(2)
        table = table.rename(columns={"relative_change": "change_%"})
        if table["drift"].any():
            st.warning("Drift detected in: " + ", ".join(table[table["drift"]]["Feature"]))
        else:
            st.success("No data drift detected (all changes are below 10%).")
        st.dataframe(table)

    st.caption("Experiment runs are logged to the `mlruns` folder. "
               "View them with: `mlflow ui`")

st.divider()
st.caption("RetailPulse | AI-Powered Retail Analytics")
