"""
app.py — Railway Forecast & Risk Dashboard
-------------------------------------------
Run with:  streamlit run app.py

Two tabs:
  1. Dashboard   – explore historical revenue, ticket volume, delays, routes
  2. Forecast    – predict future daily revenue/volume AND the delay/
                   cancellation risk of a specific planned journey
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="Railway Forecast Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    daily = pd.read_csv(os.path.join(ART, "daily_agg.csv"), parse_dates=["Date of Journey"])
    rev_model = joblib.load(os.path.join(ART, "forecast_revenue.joblib"))
    vol_model = joblib.load(os.path.join(ART, "forecast_volume.joblib"))
    risk_model = joblib.load(os.path.join(ART, "risk_model.joblib"))
    meta = joblib.load(os.path.join(ART, "encoders.joblib"))
    full_path = os.path.join(ART, "full_data.csv")
    full = pd.read_csv(full_path, parse_dates=["Date of Journey", "Date of Purchase"]) if os.path.exists(full_path) else None
    return daily, rev_model, vol_model, risk_model, meta, full


daily, rev_model, vol_model, risk_model, meta, full = load_artifacts()
min_date = pd.to_datetime(meta["min_date"])
route_df = pd.DataFrame(meta["route_stats"])

st.title("🚆 Railway Revenue, Volume & Risk Forecasting")

# ---------------------------------------------------------------------------
# Sidebar filters (applied to the Dashboard tab)
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Filters (Dashboard tab)")

data_min = daily["Date of Journey"].min().date()
data_max = daily["Date of Journey"].max().date()
date_range = st.sidebar.date_input("Date of journey range", value=(data_min, data_max),
                                    min_value=data_min, max_value=data_max)
if isinstance(date_range, tuple) and len(date_range) == 2:
    f_start, f_end = date_range
else:
    f_start, f_end = data_min, data_max

stations_all = meta["stations"]
sel_dep = st.sidebar.multiselect("Departure station", stations_all, default=[])
sel_arr = st.sidebar.multiselect("Arrival destination", stations_all, default=[])
sel_class = st.sidebar.multiselect("Ticket class", meta["ticket_classes"], default=[])
sel_status = st.sidebar.multiselect("Journey status", ["On Time", "Delayed", "Cancelled"], default=[])

filters_active = bool(sel_dep or sel_arr or sel_class or sel_status) or (f_start, f_end) != (data_min, data_max)


def apply_filters(transactions: pd.DataFrame) -> pd.DataFrame:
    d = transactions
    d = d[(d["Date of Journey"].dt.date >= f_start) & (d["Date of Journey"].dt.date <= f_end)]
    if sel_dep:
        d = d[d["Departure Station"].isin(sel_dep)]
    if sel_arr:
        d = d[d["Arrival Destination"].isin(sel_arr)]
    if sel_class:
        d = d[d["Ticket Class"].isin(sel_class)]
    if sel_status:
        d = d[d["Journey Status"].isin(sel_status)]
    return d


# Build the filtered view: re-aggregate from full transaction data if available,
# otherwise fall back to filtering the daily-aggregate table by date only.
if full is not None:
    filtered_tx = apply_filters(full)
    filtered_daily = filtered_tx.groupby("Date of Journey").agg(
        revenue=("Price", "sum"),
        tickets=("Transaction ID", "count"),
        delayed=("Journey Status", lambda s: (s == "Delayed").sum()),
        cancelled=("Journey Status", lambda s: (s == "Cancelled").sum()),
    ).reset_index().sort_values("Date of Journey")
    if not filtered_daily.empty:
        filtered_daily["dow"] = filtered_daily["Date of Journey"].dt.dayofweek
        filtered_daily["delay_rate"] = filtered_daily["delayed"] / filtered_daily["tickets"]
        filtered_daily["cancel_rate"] = filtered_daily["cancelled"] / filtered_daily["tickets"]
else:
    filtered_tx = None
    filtered_daily = daily[(daily["Date of Journey"].dt.date >= f_start) &
                            (daily["Date of Journey"].dt.date <= f_end)]

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Forecast & Risk Tool", "📈 Details & Model Insights"])

# ===========================================================================
# TAB 1 — DASHBOARD
# ===========================================================================
with tab1:
    st.subheader("Historical performance" + (" (filtered)" if filters_active else " (Jan – Apr 2024)"))

    if filtered_daily.empty:
        st.warning("No data matches the current filters. Adjust the filters in the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total revenue", f"£{filtered_daily['revenue'].sum():,.0f}")
        c2.metric("Total tickets", f"{filtered_daily['tickets'].sum():,}")
        c3.metric("Avg delay rate", f"{filtered_daily['delay_rate'].mean()*100:.1f}%")
        c4.metric("Avg cancel rate", f"{filtered_daily['cancel_rate'].mean()*100:.1f}%")

        colA, colB = st.columns(2)
        with colA:
            fig = px.line(filtered_daily, x="Date of Journey", y="revenue", title="Daily revenue")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig = px.line(filtered_daily, x="Date of Journey", y="tickets", title="Daily ticket volume")
            st.plotly_chart(fig, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            fig = px.line(filtered_daily, x="Date of Journey", y=["delay_rate", "cancel_rate"],
                           title="Daily delay & cancellation rate")
            st.plotly_chart(fig, use_container_width=True)
        with colD:
            dow_avg = filtered_daily.groupby("dow")[["revenue", "tickets"]].mean().reset_index()
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow_avg["day"] = dow_avg["dow"].map(lambda i: dow_names[i])
            fig = px.bar(dow_avg, x="day", y="revenue", title="Avg revenue by day of week")
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Export filtered daily summary (CSV)",
            data=filtered_daily.to_csv(index=False).encode("utf-8"),
            file_name="filtered_daily_summary.csv",
            mime="text/csv",
        )
        if filtered_tx is not None:
            st.download_button(
                "⬇️ Export filtered raw transactions (CSV)",
                data=filtered_tx.to_csv(index=False).encode("utf-8"),
                file_name="filtered_transactions.csv",
                mime="text/csv",
            )

    st.subheader("Route breakdown")
    if filtered_tx is not None and filters_active:
        route_view = filtered_tx.groupby(["Departure Station", "Arrival Destination"]).agg(
            avg_price=("Price", "mean"),
            trips=("Transaction ID", "count"),
            delay_rate=("Journey Status", lambda s: (s == "Delayed").mean()),
            cancel_rate=("Journey Status", lambda s: (s == "Cancelled").mean()),
        ).reset_index()
    else:
        route_view = route_df

    if route_view.empty:
        st.info("No routes match the current filters.")
    else:
        top_n = st.slider("Show top N routes by trips", 5, 30, min(12, len(route_view)))
        route_sorted = route_view.sort_values("trips", ascending=False).head(top_n).copy()
        route_sorted["route"] = route_sorted["Departure Station"] + " → " + route_sorted["Arrival Destination"]
        fig = px.bar(route_sorted, x="route", y="trips", color="delay_rate",
                     color_continuous_scale="Reds", title="Trips per route (colored by delay rate)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            route_sorted[["route", "trips", "avg_price", "delay_rate", "cancel_rate"]]
            .rename(columns={"avg_price": "avg_price(£)"})
            .style.format({"avg_price(£)": "{:.1f}", "delay_rate": "{:.1%}", "cancel_rate": "{:.1%}"}),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Export route breakdown (CSV)",
            data=route_sorted.to_csv(index=False).encode("utf-8"),
            file_name="route_breakdown.csv",
            mime="text/csv",
        )

# ===========================================================================
# TAB 2 — FORECAST & RISK TOOL
# ===========================================================================
with tab2:
    st.subheader("Forecast future revenue & ticket volume")

    horizon = st.slider("Days to forecast ahead", 7, 90, 30)
    last_date = daily["Date of Journey"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    def recursive_forecast(model, target_history, dates, min_date):
        """Roll forward day-by-day, feeding each prediction back in as lag_1,
        and updating lag_7 / 7-day rolling mean as we go (since the model was
        trained with these lag features)."""
        history = list(target_history)  # most recent values, oldest first
        preds = []
        for d in dates:
            day_index = (d - min_date).days
            dow = d.dayofweek
            month = d.month
            lag_1 = history[-1]
            lag_7 = history[-7] if len(history) >= 7 else history[0]
            roll_mean_7 = np.mean(history[-7:]) if len(history) >= 7 else np.mean(history)
            row = pd.DataFrame([{
                "day_index": day_index,
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "lag_1": lag_1,
                "lag_7": lag_7,
                "roll_mean_7": roll_mean_7,
            }])
            pred = float(model.predict(row)[0])
            preds.append(pred)
            history.append(pred)
        return np.array(preds)

    recent_history = meta.get("recent_history", [])
    if recent_history:
        rev_history = [r["revenue"] for r in recent_history]
        vol_history = [r["tickets"] for r in recent_history]
    else:
        rev_history = daily["revenue"].tail(14).tolist()
        vol_history = daily["tickets"].tail(14).tolist()

    future_rev = recursive_forecast(rev_model, rev_history, future_dates, min_date)
    future_vol = recursive_forecast(vol_model, vol_history, future_dates, min_date)
    future_rev = np.clip(future_rev, 0, None)
    future_vol = np.clip(future_vol, 0, None)

    fm = meta.get("forecast_metrics", {})
    rev_std = fm.get("revenue", {}).get("resid_std", 0)
    vol_std = fm.get("tickets", {}).get("resid_std", 0)

    fc = pd.DataFrame({
        "Date": future_dates,
        "Forecast revenue": future_rev,
        "Forecast tickets": future_vol,
    })
    step = np.arange(1, len(fc) + 1)
    rev_band = 1.96 * rev_std * np.sqrt(step)
    vol_band = 1.96 * vol_std * np.sqrt(step)
    fc["Revenue lower (95%)"] = (fc["Forecast revenue"] - rev_band).clip(lower=0)
    fc["Revenue upper (95%)"] = fc["Forecast revenue"] + rev_band
    fc["Tickets lower (95%)"] = (fc["Forecast tickets"] - vol_band).clip(lower=0)
    fc["Tickets upper (95%)"] = fc["Forecast tickets"] + vol_band

    colE, colF = st.columns(2)
    with colE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["Date of Journey"], y=daily["revenue"],
                                  name="Historical", mode="lines"))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Revenue upper (95%)"],
                                  mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Revenue lower (95%)"],
                                  mode="lines", line=dict(width=0), fill="tonexty",
                                  fillcolor="rgba(99,102,241,0.15)", name="95% interval"))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Forecast revenue"],
                                  name="Forecast", mode="lines", line=dict(dash="dash")))
        fig.update_layout(title="Revenue: history + forecast (with 95% interval)")
        st.plotly_chart(fig, use_container_width=True)
    with colF:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["Date of Journey"], y=daily["tickets"],
                                  name="Historical", mode="lines"))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Tickets upper (95%)"],
                                  mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Tickets lower (95%)"],
                                  mode="lines", line=dict(width=0), fill="tonexty",
                                  fillcolor="rgba(249,115,22,0.15)", name="95% interval"))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Forecast tickets"],
                                  name="Forecast", mode="lines", line=dict(dash="dash")))
        fig.update_layout(title="Ticket volume: history + forecast (with 95% interval)")
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Total forecast revenue (next "+str(horizon)+" days)", f"£{future_rev.sum():,.0f}")
    st.dataframe(fc.style.format({"Forecast revenue": "£{:.0f}", "Forecast tickets": "{:.0f}"}),
                 use_container_width=True, height=250)
    st.download_button(
        "⬇️ Export forecast (CSV)",
        data=fc.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_next_{horizon}_days.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Predict delay/cancellation risk for a planned journey")

    stations = meta["stations"]
    colG, colH, colI = st.columns(3)
    with colG:
        dep_station = st.selectbox("Departure station", stations, index=0)
        arr_station = st.selectbox("Arrival destination", stations, index=1)
        journey_date = st.date_input("Date of journey", value=datetime.date.today())
    with colH:
        dep_hour = st.slider("Departure hour", 0, 23, 9)
        ticket_class = st.selectbox("Ticket class", meta["ticket_classes"])
        ticket_type = st.selectbox("Ticket type", meta["ticket_types"])
    with colI:
        payment = st.selectbox("Payment method", meta["payment_methods"])
        railcard = st.selectbox("Railcard", meta["railcards"])
        purchase_type = st.selectbox("Purchase type", meta["purchase_types"])
        price = st.number_input("Ticket price (£)", min_value=1, max_value=500, value=25)

    if st.button("Predict risk", type="primary"):
        encoders = meta["encoders"]

        def enc(col, val):
            le = encoders[col]
            if val in le.classes_:
                return le.transform([val])[0]
            return 0  # fallback for unseen category

        dow = pd.Timestamp(journey_date).dayofweek
        row = pd.DataFrame([{
            "Purchase Type": enc("Purchase Type", purchase_type),
            "Payment Method": enc("Payment Method", payment),
            "Railcard": enc("Railcard", railcard),
            "Ticket Class": enc("Ticket Class", ticket_class),
            "Ticket Type": enc("Ticket Type", ticket_type),
            "Departure Station": enc("Departure Station", dep_station),
            "Arrival Destination": enc("Arrival Destination", arr_station),
            "Price": price,
            "Departure Hour": dep_hour,
            "dow": dow,
        }])

        proba = risk_model.predict_proba(row)[0]
        classes = meta["classes"]  # order matches label encoder used during training
        result = dict(zip(classes, proba))

        colJ, colK = st.columns([1, 1])
        with colJ:
            fig = px.pie(values=list(result.values()), names=list(result.keys()),
                         title="Predicted outcome probability",
                         color=list(result.keys()),
                         color_discrete_map={"On Time": "green", "Delayed": "orange", "Cancelled": "red"})
            st.plotly_chart(fig, use_container_width=True)
        with colK:
            for k, v in sorted(result.items(), key=lambda x: -x[1]):
                st.metric(k, f"{v*100:.1f}%")

            route_match = route_df[(route_df["Departure Station"] == dep_station) &
                                    (route_df["Arrival Destination"] == arr_station)]
            if not route_match.empty:
                st.caption(
                    f"Historical average for this route: "
                    f"delay rate {route_match['delay_rate'].iloc[0]*100:.1f}%, "
                    f"cancel rate {route_match['cancel_rate'].iloc[0]*100:.1f}%, "
                    f"avg price £{route_match['avg_price'].iloc[0]:.1f} "
                    f"({int(route_match['trips'].iloc[0])} historical trips)."
                )
            else:
                st.caption("No historical trips found for this exact route combination.")

        result_df = pd.DataFrame([{
            "Departure": dep_station, "Arrival": arr_station, "Date": journey_date,
            "Hour": dep_hour, "Class": ticket_class, "Type": ticket_type,
            **{f"P({k})": v for k, v in result.items()}
        }])
        st.download_button(
            "⬇️ Export this prediction (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="risk_prediction.csv",
            mime="text/csv",
        )

# ===========================================================================
# TAB 3 — DETAILS & MODEL INSIGHTS
# ===========================================================================
with tab3:
    st.subheader("Model performance")

    fm = meta.get("forecast_metrics", {})
    colM, colN, colO = st.columns(3)
    with colM:
        st.metric("Risk model accuracy", f"{meta.get('risk_accuracy', 0)*100:.1f}%",
                   f"macro-F1: {meta.get('risk_f1_macro', 0):.3f}",
                   help="Holdout accuracy predicting On Time / Delayed / Cancelled (XGBoost, tuned via CV)")
    with colN:
        rev_m = fm.get("revenue", {})
        if rev_m:
            pct = rev_m["mae"] / rev_m["mean"] * 100 if rev_m["mean"] else 0
            st.metric("Revenue forecast error (MAE)", f"£{rev_m['mae']:,.0f}", f"{pct:.1f}% of avg daily revenue")
    with colO:
        vol_m = fm.get("tickets", {})
        if vol_m:
            pct = vol_m["mae"] / vol_m["mean"] * 100 if vol_m["mean"] else 0
            st.metric("Volume forecast error (MAE)", f"{vol_m['mae']:.0f} tickets", f"{pct:.1f}% of avg daily tickets")

    with st.expander("🔧 Best hyperparameters found (RandomizedSearchCV)"):
        st.write("**Risk model (XGBoost Classifier):**", meta.get("risk_best_params", {}))
        for tgt, m in fm.items():
            st.write(f"**Forecast — {tgt} (XGBoost Regressor):**", m.get("best_params", {}))

    st.divider()
    st.subheader("What drives the delay/cancellation risk prediction?")
    fi = meta.get("feature_importance", {})
    if fi:
        fi_df = pd.DataFrame(sorted(fi.items(), key=lambda x: -x[1]), columns=["Feature", "Importance"])
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", template="plotly_white",
                     title="Risk model feature importance", color="Importance", color_continuous_scale="Blues")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher importance means the model relies more on that feature to distinguish "
            "On Time / Delayed / Cancelled outcomes."
        )

    st.divider()
    st.subheader("Delay reasons (historical)")
    reason_counts = meta.get("reason_counts", {})
    reason_df = pd.DataFrame(
        [(k, v) for k, v in reason_counts.items() if k not in ("N/A", "nan", None)],
        columns=["Reason", "Count"]
    ).sort_values("Count", ascending=False)
    if not reason_df.empty:
        fig = px.bar(reason_df, x="Reason", y="Count", template="plotly_white",
                     title="Delay reasons across all historical journeys",
                     color="Count", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No delay reason data available.")

    st.divider()
    st.subheader("Route risk map")
    fig = px.scatter(route_df, x="avg_price", y="delay_rate", size="trips", color="cancel_rate",
                      hover_data=["Departure Station", "Arrival Destination"],
                      template="plotly_white", color_continuous_scale="Reds",
                      title="Avg price vs delay rate (bubble size = trips, color = cancel rate)",
                      labels={"avg_price": "Avg Price (£)", "delay_rate": "Delay Rate"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Day-of-week × Status breakdown")
    if full is not None:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        tmp = full.copy()
        tmp["dow"] = tmp["Date of Journey"].dt.dayofweek
        cross = tmp.groupby(["dow", "Journey Status"]).size().reset_index(name="Count")
        cross["Day"] = cross["dow"].map(lambda i: dow_names[i])
        fig = px.density_heatmap(cross, x="Day", y="Journey Status", z="Count",
                                  category_orders={"Day": dow_names}, template="plotly_white",
                                  title="Journey status volume by day of week", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Models: XGBoost Regressor with cyclical date encoding + lag/rolling features (forecasts), "
        "tuned via TimeSeriesSplit RandomizedSearchCV. "
        "XGBoost Classifier with class-balanced sample weights (risk), "
        "tuned via StratifiedKFold RandomizedSearchCV. "
        "Retrain by running train_models.py on an updated CSV."
    )

st.sidebar.divider()
st.sidebar.header("About")
st.sidebar.write(
    "This app uses historical railway transaction data to:\n\n"
    "- **Dashboard**: explore revenue, volume, delays and routes — "
    "use the filters above to narrow by date, station, ticket class, or status, "
    "and export any view as CSV\n"
    "- **Forecast**: project future daily revenue/ticket volume "
    "(tuned XGBoost Regressor, recursive multi-step with lag features) — exportable\n"
    "- **Risk tool**: predict On Time / Delayed / Cancelled probability "
    "for a planned journey (tuned XGBoost Classifier) — exportable"
)
