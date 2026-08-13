"""Silo projection, reorder points and policy backtesting.

A daily `(s, S)` policy driven by the demand forecast:

    s (reorder point) = forecast demand over the risk period + safety stock(site)
    S (order-up-to)   = s + one week of forecast demand, capped at silo capacity

The simulation runs **daily**, not weekly. Four of the thirty sites hold less than one
week of demand (SITE_010 has a 158 t silo against 211 t/week), and the recorded data
shows deliveries on 98% of site-days — about 6.5 per week. A weekly single-delivery
model cannot serve those sites and understates readiness for all of them.

Safety stock is set **per site** from that site's own forecast error. Error varies 23x
across the estate (2.6 to 60.0 t per week), so one global figure both wastes capacity
at steady sites and leaves volatile ones exposed.

Validated in `NOTEBOOKS/05_Inventory_Simulation.ipynb`: 99.8% pour readiness against
45.3% for recorded practice, on 21% less cement ordered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mig_cement.config import settings

REVIEW_DAYS = 1          # stock checked daily
SHELF_LIFE_DAYS = 84     # ~12 weeks for cement in a dry silo - domain knowledge
WARMUP_DAYS = 7          # excluded from scoring; see `summarise`
BAND_LOW, BAND_HIGH = 0.20, 0.80      # workable stock band
TARGET_POUR_READINESS = 0.98


# --------------------------------------------------------------------------- #
# policy parameters
# --------------------------------------------------------------------------- #
def safety_stock(sigma_weekly: float | pd.Series,
                 lead_time_days: int | None = None,
                 z: float | None = None) -> float | pd.Series:
    """z * sigma * sqrt(lead time + review period).

    The review period matters. With weekly review and a 3-day lead time the exposure
    is 10 days, not 3 - omitting it collapses readiness to 94%. Here review is daily,
    so the risk period is lead + 1.
    """
    lead = lead_time_days if lead_time_days is not None else settings.lead_time_days
    zz = z if z is not None else settings.service_level_z
    risk_period = lead + REVIEW_DAYS
    return zz * (sigma_weekly / np.sqrt(7)) * np.sqrt(risk_period)


def reorder_point(forecast_daily: float, safety: float, capacity: float,
                  lead_time_days: int | None = None) -> float:
    """Expected demand over the risk period plus safety stock, capped at capacity."""
    lead = lead_time_days if lead_time_days is not None else settings.lead_time_days
    return float(min(forecast_daily * (lead + REVIEW_DAYS) + safety, capacity))


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def simulate(daily: pd.DataFrame,
             safety_by_site: pd.Series,
             lead_time_days: int | None = None) -> pd.DataFrame:
    """Daily (s, S) simulation, one silo per site.

    Args:
        daily: one row per site-day with `date`, `site_id`, `consumed_tonnes`,
            `planned_pour_tonnes`, `silo_capacity`, `opening_inventory_tonnes`,
            `day_forecast` and `week_forecast`.
        safety_by_site: safety stock in tonnes, indexed by `site_id`.

    Returns one row per simulated site-day. FIFO layers track stock age so anything
    held past the shelf life is written off.
    """
    lead = lead_time_days if lead_time_days is not None else settings.lead_time_days
    risk_period = lead + REVIEW_DAYS
    default_safety = float(safety_by_site.median())
    rows = []

    for site, g in daily.groupby("site_id"):
        g = g.sort_values("date").reset_index(drop=True)
        capacity = float(g.silo_capacity.iloc[0])
        safety = float(safety_by_site.get(site, default_safety))
        layers = [[float(g.opening_inventory_tonnes.iloc[0]), 0]]   # [tonnes, age_days]
        in_transit: dict[int, float] = {}

        for i, r in g.iterrows():
            arriving = in_transit.pop(i, 0.0)
            opening = sum(t for t, _ in layers)

            received = float(np.clip(arriving, 0, max(capacity - opening, 0)))
            rejected = max(arriving - received, 0.0)
            if received > 0:
                layers.append([received, 0])

            available = opening + received
            demand = float(r.consumed_tonnes)
            served = min(demand, available)

            remaining = served
            for layer in layers:
                take = min(layer[0], remaining)
                layer[0] -= take
                remaining -= take
                if remaining <= 1e-9:
                    break

            for layer in layers:
                layer[1] += 1
            expired = sum(t for t, age in layers if age > SHELF_LIFE_DAYS)
            layers = [[t, a] for t, a in layers if a <= SHELF_LIFE_DAYS and t > 1e-9]
            closing = sum(t for t, _ in layers)

            # order against the inventory position, so nothing is double-ordered
            position = closing + sum(in_transit.values())
            s = r.day_forecast * risk_period + safety
            S = min(s + r.week_forecast, capacity)
            order = max(S - position, 0.0) if position < s else 0.0
            if order > 1e-6:
                arrival = i + lead
                in_transit[arrival] = in_transit.get(arrival, 0.0) + order

            rows.append({
                "site_id": site, "date": r.date, "opening": opening,
                "received": received, "rejected": rejected, "demand": demand,
                "served": served, "shortfall": demand - served, "closing": closing,
                "expired": expired, "order": order, "capacity": capacity,
                "reorder_point": s, "order_up_to": S,
                "pour_day": r.planned_pour_tonnes > 0,
                "pour_met": (demand - served) <= 1e-6,
                "utilisation": closing / capacity,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def summarise(sim: pd.DataFrame, warmup_days: int = WARMUP_DAYS) -> pd.Series:
    """Policy metrics, scored on pour days after a warm-up.

    Two exclusions, both standard for an inventory backtest:
      * pour days only - a day with no scheduled pour cannot cause a stockout
      * after a warm-up - the simulation starts with nothing in transit, so the first
        lead-time window cannot be replenished. 41 of 43 stockouts fall in week 1,
        and they correlate -0.80 with opening stock: an initialisation artefact.
    """
    scored = sim[sim.date >= sim.date.min() + pd.Timedelta(days=warmup_days)]
    pour = scored[scored.pour_day]
    return pd.Series({
        "pour_readiness": pour.pour_met.mean(),
        "stockout_days": int((~pour.pour_met).sum()),
        "unmet_tonnes": scored.shortfall.sum(),
        "mean_utilisation": scored.utilisation.mean(),
        "write_offs_t": scored.expired.sum() + scored.rejected.sum(),
        "total_ordered_t": scored.order.sum(),
        "deliveries": int((scored.received > 0).sum()),
        "sites_meeting_target": int(
            (pour.groupby("site_id").pour_met.mean() >= TARGET_POUR_READINESS).sum()),
    })


def efficiency(utilisation) -> dict[str, float]:
    """Four readings of 'inventory utilisation efficiency'.

    The brief names both stockouts and overstocking as failures, so mean fill alone
    is inadequate - it rewards hoarding. The share of days in a workable band is the
    recommended measure.
    """
    u = np.asarray(utilisation, dtype=float)
    return {
        "mean_fill_pct": 100 * u.mean(),
        "pct_days_in_band": 100 * ((u >= BAND_LOW) & (u <= BAND_HIGH)).mean(),
        "pct_days_jammed_or_starved": 100 * ((u > 0.90) | (u < 0.10)).mean(),
        "dispersion_std": u.std(),
    }


# --------------------------------------------------------------------------- #
# operational output
# --------------------------------------------------------------------------- #
def reorder_alerts(sim: pd.DataFrame, as_of: pd.Timestamp | None = None
                   ) -> pd.DataFrame:
    """Current reorder position per site, for the dashboard alert table.

    Severity is deliberately not colour-only - site tablets get read in bad light.
        red    projected to stock out before the next delivery lands
        amber  below the reorder point
        green  healthy
    """
    as_of = as_of or sim.date.max()
    latest = (sim[sim.date <= as_of].sort_values("date")
              .groupby("site_id").last().reset_index())

    recent = (sim[sim.date <= as_of]
              .groupby("site_id").demand
              .apply(lambda s: s.tail(7).mean()).rename("mean_daily_demand"))
    out = latest.merge(recent, on="site_id")

    out["days_to_stockout"] = np.where(
        out.mean_daily_demand > 0, out.closing / out.mean_daily_demand, np.inf)
    out["below_reorder_point"] = out.closing < out.reorder_point
    out["suggested_order_t"] = (out.order_up_to - out.closing).clip(lower=0).round(1)

    out["severity"] = np.select(
        [out.days_to_stockout < settings.lead_time_days, out.below_reorder_point],
        ["red", "amber"], default="green")

    cols = ["site_id", "date", "closing", "capacity", "utilisation",
            "mean_daily_demand", "days_to_stockout", "reorder_point",
            "suggested_order_t", "severity"]
    return (out[cols].sort_values("days_to_stockout")
            .reset_index(drop=True).round(2))


def project_silo_levels(sim: pd.DataFrame, site_id: str) -> pd.DataFrame:
    """Stock trajectory for one site — the dashboard's primary chart.

    Returns closing stock alongside capacity and the reorder point, so both can be
    drawn as reference lines and the stockout crossing is visible.
    """
    s = sim[sim.site_id == site_id].sort_values("date")
    return s[["date", "opening", "received", "demand", "served", "closing",
              "capacity", "reorder_point", "shortfall", "pour_met"]].reset_index(drop=True)


def backtest_policy(sim: pd.DataFrame, actual: pd.DataFrame,
                    warmup_days: int = WARMUP_DAYS) -> pd.DataFrame:
    """Policy against MIG's recorded practice.

    `actual` is the repaired ledger over the same window - the recorded deliveries,
    not a modelled counterfactual. The brief describes current practice as manual
    estimator projections against a rolling 4-week schedule, so there is nothing to
    simulate: what MIG did is in the data.
    """
    start = sim.date.min() + pd.Timedelta(days=warmup_days)
    pol = sim[sim.date >= start]
    act = actual[(actual.date >= start) & (actual.date <= sim.date.max())].copy()
    act["utilisation"] = act.closing_inventory_tonnes / act.silo_capacity
    act_pour = act[act.planned_pour_tonnes > 0]
    pol_pour = pol[pol.pour_day]

    act_band = ((act.utilisation >= BAND_LOW) & (act.utilisation <= BAND_HIGH)).mean()
    pol_band = ((pol.utilisation >= BAND_LOW) & (pol.utilisation <= BAND_HIGH)).mean()
    act_ready = (act_pour.served_tonnes >= act_pour.planned_pour_tonnes - 1e-6).mean()
    pol_ready = pol_pour.pour_met.mean()
    act_waste = act.rejected_delivery_tonnes.sum()
    pol_waste = pol.rejected.sum() + pol.expired.sum()

    rows = [
        ("1. Pour readiness >= 98%", f"{100*act_ready:.1f}%", f"{100*pol_ready:.1f}%",
         f"{100*(pol_ready-act_ready):+.1f} pp", pol_ready >= TARGET_POUR_READINESS),
        ("2. Inventory utilisation +20%", f"{100*act_band:.1f}%", f"{100*pol_band:.1f}%",
         f"{100*(pol_band/act_band-1):+.0f}%", (pol_band/act_band - 1) >= 0.20),
        ("3. Write-offs -30%", f"{act_waste:,.0f} t", f"{pol_waste:,.0f} t",
         f"{100*(pol_waste/act_waste-1):+.0f}%" if act_waste else "n/a",
         bool(act_waste and (pol_waste/act_waste - 1) <= -0.30)),
    ]
    out = pd.DataFrame(rows, columns=["target", "actual_practice", "forecast_policy",
                                      "change", "met"]).set_index("target")
    out["met"] = np.where(out.met, "MET", "NOT MET")
    return out
