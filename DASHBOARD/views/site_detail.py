"""Site drill-down — the silo projection chart is the primary view of the product."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import theme

import data as dat


def render() -> None:
    sites = dat.get_sites()
    site = st.selectbox("Site", sites, index=0)

    proj = dat.get_site_projection(site)
    fc = dat.get_forecasts()
    site_fc = fc[fc.site_id == site]
    alerts = dat.get_alerts()
    alert = alerts[alerts.site_id == site]

    if proj.empty:
        st.info("No simulation available. Run `make pipeline`.")
        return

    capacity = float(proj.capacity.iloc[0])

    # --- site KPIs --------------------------------------------------------- #
    # st.metric renders its delta as a grey arrow regardless of meaning, so a
    # severity of "green" read as neutral. These cards colour by status instead.
    fill = proj.closing.iloc[-1] / capacity
    severity = alert.severity.iloc[0] if not alert.empty else None
    status = theme.SEVERITY_STATUS.get(severity, "neutral")

    mean_weekly = site_fc.actual_tonnes.mean() if not site_fc.empty else float("nan")
    cover = capacity / mean_weekly if mean_weekly else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.kpi_card("Silo capacity", f"{capacity:,.0f} t",
                       "—" if cover != cover else f"{cover:.1f}x weekly demand",
                       "neutral" if cover != cover
                       else theme.status_for(cover, 1.0, lower_is_better=False))
    with c2:
        theme.kpi_card("Current stock", f"{proj.closing.iloc[-1]:,.1f} t",
                       f"{100*fill:.0f}% full",
                       "pass" if 0.2 <= fill <= 0.8 else "warn")
    if not alert.empty:
        days = alert.days_to_stockout.iloc[0]
        with c3:
            theme.kpi_card("Days to stockout",
                           "—" if days == float("inf") else f"{days:.1f}",
                           theme.SEVERITY_LABEL.get(severity, ""), status)
        with c4:
            theme.kpi_card("Suggested order",
                           f"{alert.suggested_order_t.iloc[0]:,.0f} t",
                           "order now" if severity == "red" else
                           "order soon" if severity == "amber" else "no action",
                           status)

    missed = int((~proj.pour_met).sum())
    if missed:
        st.warning(f"{missed} day(s) where demand exceeded available stock. "
                   "Days in the first week are a simulation warm-up artefact — "
                   "the run starts with nothing in transit.")

    st.divider()

    # --- the primary chart -------------------------------------------------- #
    st.markdown("**Projected silo level**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=proj.date, y=proj.closing, name="Stock",
                             mode="lines", line=dict(width=3, color=theme.PASS),
                             fill="tozeroy", fillcolor="rgba(26,127,55,0.12)"))
    fig.add_trace(go.Scatter(x=proj.date, y=proj.reorder_point,
                             name="Reorder point", mode="lines",
                             line=dict(width=2, dash="dot", color=theme.WARN)))
    fig.add_hline(y=capacity, line=dict(color=theme.FAIL, dash="dash"),
                  annotation_text=f"capacity {capacity:,.0f} t",
                  annotation_position="top right")

    stockouts = proj[proj.shortfall > 1e-6]
    if not stockouts.empty:
        fig.add_trace(go.Scatter(x=stockouts.date, y=[0] * len(stockouts),
                                 name="Stockout", mode="markers",
                                 marker=dict(size=11, symbol="x", color="#cf222e")))

    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="tonnes", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Where the stock line reaches zero on a pour day, the pour could not "
               "be served. The reorder point rises and falls with forecast demand.")

    # --- forecast accuracy -------------------------------------------------- #
    st.markdown("**Weekly demand: forecast vs actual**")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=site_fc.date, y=site_fc.actual_tonnes, name="Actual",
                          marker_color=theme.CHART_ACTUAL))
    fig2.add_trace(go.Scatter(x=site_fc.date, y=site_fc.forecast_tonnes,
                              name="Forecast", mode="lines+markers",
                              line=dict(width=3, color=theme.CHART_FORECAST)))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis_title="tonnes", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Site MAPE", f"{100*site_fc.pct_error.mean():.1f}%")
    b.metric("Mean weekly demand", f"{site_fc.actual_tonnes.mean():,.0f} t")
    c.metric("Capacity / weekly demand",
             f"{capacity/site_fc.actual_tonnes.mean():.2f}x")

    if capacity / site_fc.actual_tonnes.mean() < 1.0:
        st.error("This silo holds less than one week of demand. No ordering policy "
                 "can protect it against a forecast miss — it is a capacity "
                 "constraint, not a forecasting one.")

    with st.expander("Daily detail"):
        st.dataframe(proj.round(2), use_container_width=True, height=320)
