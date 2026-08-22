"""Model performance — for the analyst, not the site manager."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data as dat
import theme

TARGET_MAPE = 0.15


ALL = "All sites"


def _metrics(fc) -> dict:
    """Scored the same way the pipeline scores the hold-out, so a subset is
    directly comparable with the estate-wide figures in the model card."""
    return {"MAPE": fc.pct_error.mean(),
            "RMSE": float(np.sqrt((fc.abs_error ** 2).mean())),
            "bias": float((fc.forecast_tonnes - fc.actual_tonnes).mean())}


def render() -> None:
    fc_all = dat.get_forecasts()
    meta = dat.get_model_metadata()

    if fc_all.empty:
        st.info("No forecasts available. Run `make pipeline`.")
        return

    # --- drill-down --------------------------------------------------------- #
    site = st.selectbox("Site", [ALL] + dat.get_sites(), index=0)
    is_all = site == ALL
    fc = fc_all if is_all else fc_all[fc_all.site_id == site]

    if fc.empty:
        st.info(f"No hold-out weeks for {site}.")
        return

    m = _metrics(fc)
    estate = _metrics(fc_all)
    hold_mape, bias = m["MAPE"], m["bias"]
    val = meta.get("validation_metrics", {})
    gap = hold_mape - val.get("MAPE", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.kpi_card("Hold-out MAPE", f"{100*hold_mape:.2f}%",
                       "target ≤ 15% — met" if hold_mape <= TARGET_MAPE
                       else "target ≤ 15% — missed",
                       theme.status_for(hold_mape, TARGET_MAPE, lower_is_better=True))
    with c2:
        theme.kpi_card("RMSE", f"{m['RMSE']:.1f} t", "per site-week", "neutral")
    with c3:
        theme.kpi_card("Bias", f"{bias:+.2f} t",
                       "near zero" if abs(bias) < 3 else "systematic drift",
                       "pass" if abs(bias) < 3 else "warn")
    with c4:
        if is_all:
            theme.kpi_card("Validation → test", f"{100*gap:+.2f} pp",
                           "small gap" if gap < 0.03 else "large gap",
                           "pass" if gap < 0.03 else "warn")
        else:
            delta = hold_mape - estate["MAPE"]
            theme.kpi_card("vs estate", f"{100*delta:+.2f} pp",
                           "better than average" if delta < 0 else "worse than average",
                           "pass" if delta < 0 else "warn")

    if is_all:
        st.caption(f"{len(fc)} site-weeks across {fc.site_id.nunique()} sites. The "
                   "validation-to-test gap is the cost of decisions made against the "
                   "validation window. RMSE barely moved, so the loss sits in "
                   "low-volume weeks rather than in general accuracy.")
    else:
        st.caption(f"{len(fc)} hold-out weeks for {site}, scored the same way as the "
                   f"estate. Estate MAPE is {100*estate['MAPE']:.2f}%.")

    st.divider()
    left, right = st.columns(2)

    # --- accuracy across the horizon --------------------------------------- #
    with left:
        st.markdown("**Accuracy by forecast week**")
        by_h = fc.groupby("horizon_week").agg(
            mape=("pct_error", "mean"),
            rmse=("abs_error", lambda s: np.sqrt((s ** 2).mean()))).reset_index()
        fig = go.Figure()
        bar_colours = [theme.PASS if m <= TARGET_MAPE else theme.FAIL
                       for m in by_h.mape]
        fig.add_trace(go.Bar(x=by_h.horizon_week, y=100 * by_h.mape, name="MAPE %",
                             marker_color=bar_colours))
        fig.add_hline(y=100 * TARGET_MAPE, line=dict(color=theme.FAIL, dash="dash"),
                      annotation_text="15% target")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="weeks ahead", yaxis_title="MAPE %")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("No decay across the horizon — the features are schedule-derived "
                   "rather than recursive, so week 8 is no harder than week 1."
                   if is_all else
                   "One observation per bar at site level, so read the shape as "
                   "indicative rather than as a trend.")

    # --- forecast vs actual -------------------------------------------------- #
    with right:
        st.markdown("**Forecast vs actual**")
        lim = float(max(fc.actual_tonnes.max(), fc.forecast_tonnes.max())) * 1.05
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc.actual_tonnes, y=fc.forecast_tonnes,
                                 mode="markers", name="site-weeks",
                                 marker=dict(size=6, opacity=0.55,
                                             color=theme.CHART_ACTUAL)))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                 name="perfect",
                                 line=dict(dash="dash", color=theme.NEUTRAL)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="actual (t)", yaxis_title="forecast (t)",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- the table: sites when aggregated, weeks when drilled in ------------- #
    if is_all:
        st.markdown("**Accuracy by site**")
        by_site = (fc.groupby("site_id")
                   .agg(mean_actual=("actual_tonnes", "mean"),
                        mape=("pct_error", "mean"),
                        rmse=("abs_error", lambda s: np.sqrt((s ** 2).mean())),
                        bias=("forecast_tonnes", "mean"))
                   .assign(bias=lambda d: d.bias
                           - fc.groupby("site_id").actual_tonnes.mean())
                   .sort_values("mape", ascending=False))
        by_site["meets 15%"] = np.where(by_site.mape <= TARGET_MAPE, "yes", "no")

        n_ok = int((by_site.mape <= TARGET_MAPE).sum())
        st.caption(f"{n_ok} of {len(by_site)} sites individually meet the 15% target. "
                   "The aggregate passes because the larger sites do — sites above the "
                   "threshold are where the forecast is least reliable. Select a site "
                   "above to drill into its weeks.")
        st.dataframe(by_site.round(3), use_container_width=True, height=340)
    else:
        st.markdown(f"**Week by week — {site}**")
        weeks = (fc[["date", "horizon_week", "actual_tonnes", "forecast_tonnes",
                     "abs_error", "pct_error"]]
                 .assign(bias=fc.forecast_tonnes - fc.actual_tonnes)
                 .sort_values("horizon_week"))
        weeks["meets 15%"] = np.where(weeks.pct_error <= TARGET_MAPE, "yes", "no")
        n_ok = int((weeks.pct_error <= TARGET_MAPE).sum())
        st.caption(f"{n_ok} of {len(weeks)} hold-out weeks land inside 15%.")
        st.dataframe(weeks.round(3), use_container_width=True, hide_index=True)

    with st.expander("Model card"):
        st.json({k: meta.get(k) for k in
                 ["name", "estimator", "features", "grain", "horizon_weeks",
                  "target", "trained_on", "project_target", "versions"]
                 if k in meta})
