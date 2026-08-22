"""Reorder alerts — the page operations actually act on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as dat
import theme

SEVERITY_ICON = {"red": "🔴 ORDER NOW", "amber": "🟠 Order soon", "green": "🟢 OK"}


def render() -> None:
    alerts = dat.get_alerts()
    if alerts.empty:
        st.info("No simulation available. Run `make pipeline`.")
        return

    panel = dat.get_weekly_panel()
    region_of = panel.groupby("site_id").region.first()
    alerts = alerts.assign(region=alerts.site_id.map(region_of))

    # --- filter ------------------------------------------------------------ #
    # Severity is a property of the site, not a view setting. Hiding a red site
    # behind a filter is exactly how a stockout gets missed, so every site is
    # always listed and severity is carried in the row instead.
    all_regions = sorted(alerts.region.dropna().unique())
    regions = st.multiselect("Region", all_regions, default=all_regions)
    view = alerts[alerts.region.isin(regions)]

    counts = view.severity.value_counts()
    c1, c2, c3 = st.columns(3)
    for col, sev in zip((c1, c2, c3), ("red", "amber", "green"), strict=True):
        with col:
            theme.kpi_card(theme.SEVERITY_LABEL[sev], str(int(counts.get(sev, 0))),
                           "sites", theme.SEVERITY_STATUS[sev])

    st.caption(f"All {len(view)} sites in the selected regions, "
               "sorted by days to stockout.")

    # --- the table --------------------------------------------------------- #
    table = view.copy()
    table["status"] = table.severity.map(SEVERITY_ICON)
    table["days_to_stockout"] = table.days_to_stockout.replace(
        [np.inf, -np.inf], np.nan)
    # utilisation is a 0-1 fraction; ProgressColumn formats the raw value, so a
    # 5% fill printed as "%.0f%%" showed 0%. Scale to 0-100 for display.
    table["fill_pct"] = (100 * table.utilisation).round(0)

    table = table[["status", "severity", "site_id", "region", "closing", "capacity",
                   "fill_pct", "days_to_stockout", "reorder_point",
                   "suggested_order_t"]].rename(columns={
        "closing": "stock (t)", "capacity": "capacity (t)",
        "fill_pct": "fill %", "days_to_stockout": "days to stockout",
        "reorder_point": "reorder at (t)", "suggested_order_t": "order (t)"})

    st.dataframe(
        theme.style_alerts_table(table).format({
            "stock (t)": "{:.1f}", "capacity (t)": "{:.0f}", "fill %": "{:.0f}%",
            "days to stockout": "{:.1f}", "reorder at (t)": "{:.0f}",
            "order (t)": "{:.0f}"}),
        use_container_width=True, hide_index=True, height=430,
        column_order=["status", "site_id", "region", "stock (t)", "capacity (t)",
                      "fill %", "days to stockout", "reorder at (t)", "order (t)"])
    st.caption("Rows are tinted by urgency: red needs an order now, amber is below "
               "the reorder point, green is healthy.")

    st.download_button(
        "Download as CSV", view.to_csv(index=False).encode(),
        file_name="reorder_alerts.csv", mime="text/csv")

    st.divider()

    # --- utilisation heatmap ------------------------------------------------ #
    st.markdown("**Silo utilisation by site and week**")
    sim = dat.get_simulation()
    sim = sim.assign(week=sim.date.dt.to_period("W-SUN").dt.start_time)
    grid = (sim[sim.site_id.isin(view.site_id)]
            .pivot_table(index="site_id", columns="week",
                         values="utilisation", aggfunc="mean"))

    if grid.empty:
        st.info("No sites match the current filters.")
        return

    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[d.date() for d in grid.columns], y=grid.index,
        colorscale=theme.FILL_SCALE,
        zmin=0, zmax=1, colorbar=dict(title="fill")))
    fig.update_layout(height=max(320, 22 * len(grid)),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red at both ends: an empty silo risks a stockout, a full one cannot "
               "accept a delivery. Green is the workable 20–80% band.")

    # --- efficiency --------------------------------------------------------- #
    from mig_cement.inventory import simulate as inv
    scored = sim[sim.date >= sim.date.min() + pd.Timedelta(days=inv.WARMUP_DAYS)]
    eff = inv.efficiency(scored.utilisation)
    a, b, c = st.columns(3)
    a.metric("Mean fill", f"{eff['mean_fill_pct']:.1f}%")
    b.metric("Days in workable band", f"{eff['pct_days_in_band']:.1f}%")
    c.metric("Days jammed or starved", f"{eff['pct_days_jammed_or_starved']:.1f}%")
