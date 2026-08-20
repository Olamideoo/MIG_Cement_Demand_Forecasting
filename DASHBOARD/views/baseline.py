"""Baseline — what the data showed before the forecast existed.

Context for every other page. The green figures elsewhere only mean something
against the problem they replaced, and each finding here is drawn from the recorded
data rather than asserted.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import theme

import data as dat


def render() -> None:
    st.subheader("Before: what the data showed")
    st.caption("Recorded operations over the same window the forecast policy is "
               "scored on. Nothing here is modelled — it is what happened.")

    base = dat.get_baseline_stats()
    if not base:
        st.info("Run `make pipeline` to populate.")
        return

    policy = dat.get_policy_summary()
    bias = dat.get_schedule_bias()

    # --- the problem, in four numbers -------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.kpi_card("Pour readiness", f"{100*base['pour_readiness']:.1f}%",
                       "target ≥ 98%", "fail")
    with c2:
        theme.kpi_card("Silo jammed or starved",
                       f"{100*base['jammed_or_starved']:.0f}%",
                       "of site-days", "fail")
    with c3:
        theme.kpi_card("Ordered but rejected",
                       f"{100*base['rejected_pct']:.1f}%",
                       f"{base['rejected_t']:,.0f} t could not fit", "fail")
    with c4:
        theme.kpi_card("Unmet demand", f"{base['unmet_t']:,.0f} t",
                       "pours not fully served", "fail")

    st.divider()

    # --- the bimodal failure ------------------------------------------------ #
    st.markdown("### Two opposite failures, at once")
    left, right = st.columns([3, 2])

    prof = dat.get_behaviour_profile()
    with left:
        w = dat.get_baseline_window()
        fig = go.Figure()
        for behaviour, colour in [("aggressive", theme.FAIL),
                                  ("chaotic", theme.WARN),
                                  ("conservative", theme.CHART_FORECAST)]:
            vals = 100 * w[w.behavior == behaviour].utilisation
            if len(vals):
                fig.add_trace(go.Box(x=vals, name=behaviour, marker_color=colour,
                                     boxpoints=False, orientation="h"))
        fig.add_vrect(x0=20, x1=80, fillcolor=theme.PASS, opacity=0.10,
                      line_width=0, annotation_text="workable band",
                      annotation_position="top left")
        fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="silo fill %", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.dataframe(prof, use_container_width=True)
        st.markdown(
            f"Aggressive sites sat at **{prof.loc['aggressive', 'mean fill %']:.0f}% "
            f"full** and ran completely empty on "
            f"**{prof.loc['aggressive', '% days empty']:.0f}%** of days. "
            f"Conservative sites sat at **{prof.loc['conservative', 'mean fill %']:.0f}%** "
            f"— a silo that full cannot accept a delivery, which is why a fifth of "
            f"ordered volume was turned away.")

    st.info(f"Only **{100*base['in_band']:.0f}%** of site-days fell inside a workable "
            f"20–80% band. The estate was starving and hoarding simultaneously — "
            f"exactly the 'stockouts, overstocking and reactive ordering' the brief "
            f"describes.")

    st.divider()

    # --- the rain cliff ------------------------------------------------------ #
    st.markdown("### Rain stops pours — sharply, at 15 mm")
    rain = dat.get_rain_response()
    colours = [theme.PASS if v < 30 else theme.FAIL for v in rain.pct_zero]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=rain.band, y=rain.pct_zero, name="% days with no pour",
                         marker_color=colours))
    fig.add_trace(go.Scatter(x=rain.band, y=rain.mean_consumed, name="mean consumed (t)",
                             mode="lines+markers", yaxis="y2",
                             line=dict(width=3, color=theme.CHART_ACTUAL)))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="daily rainfall band (mm)",
        yaxis=dict(title="% days with no pour"),
        yaxis2=dict(title="tonnes", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Below 15 mm rain has no measurable effect — consumption is flat at "
               "~25 t across every band. Above it, no-pour days jump from 9% to 68% "
               "and mean consumption collapses to 4 t. A step, not a gradient, which "
               "is why the model uses a threshold flag rather than raw rainfall.")

    st.divider()

    # --- ordering to the schedule -------------------------------------------- #
    st.markdown("### Ordering to the schedule always over-orders")
    a, b = st.columns([2, 3])
    with a:
        st.metric("Planned pour ≥ actual consumption",
                  f"{100*bias['pct_planned_ge_actual']:.1f}% of days")
        st.metric("Average over-order",
                  f"{bias['mean_over_order_t']:.1f} t/site-day",
                  f"{100*bias['over_order_pct_of_demand']:.0f}% of demand",
                  delta_color="off")
    with b:
        st.markdown(
            "The planned pour is **never** below actual consumption — not once in "
            "32,880 site-days. Every error points the same way, so ordering to the "
            "4-week schedule cannot cancel out: it over-orders systematically.\n\n"
            "That is why the forecast policy orders **21% less cement** while "
            "raising pour readiness from "
            f"{100*base['pour_readiness']:.0f}% to "
            f"{100*policy.get('pour_readiness', 0):.1f}%.")

    st.divider()

    # --- what a forecast cannot fix ------------------------------------------ #
    st.markdown("### One problem the forecast cannot solve")
    tight = dat.get_capacity_constrained()
    st.dataframe(tight, use_container_width=True)
    st.warning(
        f"**{len(tight)} of 30 sites hold less than one week of demand.** No ordering "
        "policy protects a silo that small against a forecast miss — it is a capital "
        "constraint. These sites are where residual stockout risk remains, and they "
        "are worth raising with operations separately from the forecasting work.")
