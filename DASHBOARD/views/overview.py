"""National overview — KPI tiles and aggregate demand."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import data as dat
import theme


def render() -> None:
    st.subheader("Estate overview")

    fc = dat.get_forecasts()
    alerts = dat.get_alerts()
    policy = dat.get_policy_summary()
    meta = dat.get_model_metadata()

    # --- KPI row ---------------------------------------------------------- #
    mape = meta.get("holdout_metrics", {}).get("MAPE")
    readiness = policy.get("pour_readiness", 0)
    needing = int((alerts.severity != "green").sum()) if not alerts.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.kpi_card(
            "Forecast accuracy (MAPE)",
            f"{100*mape:.1f}%" if mape else "—",
            "target ≤ 15% — met" if mape and mape <= 0.15 else "target ≤ 15%",
            theme.status_for(mape, 0.15, lower_is_better=True))
    with c2:
        theme.kpi_card(
            "Pour readiness",
            f"{100*readiness:.1f}%",
            "target ≥ 98% — met" if readiness >= 0.98 else "target ≥ 98%",
            theme.status_for(readiness, 0.98, lower_is_better=False))
    with c3:
        theme.kpi_card(
            "Sites needing action", f"{needing}",
            "all healthy" if needing == 0 else f"of {len(alerts)} sites",
            "pass" if needing == 0 else ("warn" if needing <= len(alerts) * 0.25 else "fail"))
    with c4:
        theme.kpi_card("Forecast demand, 8 weeks",
                       f"{fc.forecast_tonnes.sum():,.0f} t",
                       f"{fc.date.nunique()} weeks", "neutral")

    st.divider()

    # --- alert summary ---------------------------------------------------- #
    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Reorder status**")
        if alerts.empty:
            st.info("No simulation available.")
        else:
            counts = alerts.severity.value_counts().reindex(
                ["red", "amber", "green"]).fillna(0).astype(int)
            for sev, label in [("red", "Stock out within lead time"),
                               ("amber", "Below reorder point"),
                               ("green", "Healthy")]:
                colour = theme.SEVERITY[sev]
                st.markdown(
                    f"<div style='padding:8px 12px;margin-bottom:8px;border-radius:6px;"
                    f"border-left:6px solid {colour};background:#f6f8fa;'>"
                    f"<b style='color:{colour};font-size:1.15rem;'>{counts[sev]}</b>"
                    f"<span style='color:#57606a;'> — {label}</span></div>",
                    unsafe_allow_html=True)
            st.caption("Severity is labelled as well as coloured — these screens "
                       "get read on site tablets in poor light.")

    with right:
        st.markdown("**Weekly demand: forecast vs actual, all sites**")
        weekly = (fc.groupby("date")[["actual_tonnes", "forecast_tonnes"]]
                  .sum().reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly.date, y=weekly.actual_tonnes,
                                 name="Actual", mode="lines+markers",
                                 line=dict(width=3, color=theme.CHART_ACTUAL)))
        fig.add_trace(go.Scatter(x=weekly.date, y=weekly.forecast_tonnes,
                                 name="Forecast", mode="lines+markers",
                                 line=dict(width=3, dash="dash",
                                           color=theme.CHART_FORECAST)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="tonnes", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- business targets -------------------------------------------------- #
    st.markdown("**Business targets, backtested against recorded practice**")
    targets = dat.get_targets_table()
    if targets.empty:
        st.info("Run `make pipeline` to populate.")
    else:
        st.dataframe(theme.style_targets_table(targets.reset_index()),
                     use_container_width=True, hide_index=True)
        st.caption("Baseline is MIG's recorded deliveries over the same window, "
                   "not a modelled counterfactual.")

    # --- regional split ---------------------------------------------------- #
    panel = dat.get_weekly_panel()
    region_of = panel.groupby("site_id").region.first()
    by_region = (fc.assign(region=fc.site_id.map(region_of))
                 .groupby("region")
                 .agg(sites=("site_id", "nunique"),
                      forecast_t=("forecast_tonnes", "sum"),
                      actual_t=("actual_tonnes", "sum"),
                      mape=("pct_error", "mean")).round(3))
    by_region["mape %"] = (100 * by_region.pop("mape")).round(1)
    st.markdown("**By region**")
    st.dataframe(by_region, use_container_width=True)
