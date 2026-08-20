"""Shared colour theme and status-aware KPI cards.

`st.metric` colours its delta by sign, not by whether a target is met, so a metric
where lower is better (MAPE) reads backwards. These cards take an explicit status
instead.

Colours are chosen to stay distinguishable for the most common colour-vision
deficiencies, and every status carries a **word** as well as a colour — these screens
get read on site tablets in poor light.
"""

from __future__ import annotations

import streamlit as st

PASS = "#1a7f37"      # green
WARN = "#bf8700"      # amber
FAIL = "#cf222e"      # red
NEUTRAL = "#57606a"   # grey

STATUS = {
    "pass": (PASS, "#dafbe1", "on target"),
    "warn": (WARN, "#fff8c5", "watch"),
    "fail": (FAIL, "#ffebe9", "off target"),
    "neutral": (NEUTRAL, "#f6f8fa", ""),
}

SEVERITY = {"red": FAIL, "amber": WARN, "green": PASS}
SEVERITY_LABEL = {"red": "ORDER NOW", "amber": "Order soon", "green": "healthy"}
SEVERITY_STATUS = {"red": "fail", "amber": "warn", "green": "pass"}

# diverging scale for silo fill: red at both ends - an empty silo risks a stockout,
# a full one cannot accept a delivery
FILL_SCALE = [[0.0, FAIL], [0.15, WARN], [0.35, PASS],
              [0.65, PASS], [0.85, WARN], [1.0, FAIL]]

CHART_ACTUAL = "#0969da"
CHART_FORECAST = "#8250df"


def status_for(value: float, target: float, lower_is_better: bool,
               warn_margin: float = 0.10) -> str:
    """`pass` if the target is met, `warn` if within `warn_margin` of it, else `fail`."""
    if value is None:
        return "neutral"
    if lower_is_better:
        if value <= target:
            return "pass"
        return "warn" if value <= target * (1 + warn_margin) else "fail"
    if value >= target:
        return "pass"
    return "warn" if value >= target * (1 - warn_margin) else "fail"


def kpi_card(label: str, value: str, note: str = "", status: str = "neutral") -> None:
    """A KPI tile whose colour reflects target attainment, not delta direction.

    The markup is emitted as one unbroken line. Streamlit renders it through a
    markdown parser, where a blank line closes an HTML block - so a card with no
    note used to leave its final `</div>` outside the block, and the indentation
    turned that orphan tag into a visible code block.
    """
    colour, background, default_note = STATUS.get(status, STATUS["neutral"])
    note = note or default_note
    pill = (f"<div style=\"display:inline-block;margin-top:8px;padding:2px 10px;"
            f"border-radius:999px;background:{background};color:{colour};"
            f"font-size:0.78rem;font-weight:600;\">{note}</div>") if note else ""

    html = (
        f"<div style=\"border:1px solid #d0d7de;border-left:5px solid {colour};"
        f"border-radius:8px;padding:14px 16px;background:#ffffff;height:100%;\">"
        f"<div style=\"color:{NEUTRAL};font-size:0.82rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.03em;\">{label}</div>"
        f"<div style=\"font-size:2.1rem;font-weight:700;color:{colour};"
        f"line-height:1.25;\">{value}</div>"
        f"{pill}</div>")

    st.markdown(html, unsafe_allow_html=True)


def status_badge(text: str, status: str = "neutral") -> str:
    """Inline badge, for use inside other markdown."""
    colour, background, _ = STATUS.get(status, STATUS["neutral"])
    return (f"<span style='padding:2px 10px;border-radius:999px;"
            f"background:{background};color:{colour};font-weight:600;"
            f"font-size:0.8rem;'>{text}</span>")


def style_targets_table(df):
    """Tint only the status column.

    Colouring whole rows green reads as a rubber stamp even when the results are
    real. Restricting colour to the verdict keeps the numbers legible and lets the
    reader judge them.
    """
    if "met" not in df.columns:
        return df

    def colour_status(col):
        return [f"background-color:{STATUS['pass'][1]};color:{PASS};font-weight:700"
                if str(v).upper() == "MET"
                else f"background-color:{STATUS['fail'][1]};color:{FAIL};font-weight:700"
                for v in col]

    return df.style.apply(colour_status, subset=["met"])


def style_alerts_table(df, severity_col: str = "severity"):
    """Tint each row by its reorder severity - red, amber or green."""
    if severity_col not in df.columns:
        return df

    def colour_row(row):
        sev = row.get(severity_col, "green")
        key = {"red": "fail", "amber": "warn", "green": "pass"}.get(sev, "neutral")
        colour, background, _ = STATUS[key]
        base = f"background-color:{background};"
        return [base + (f"color:{colour};font-weight:700"
                        if c == severity_col else "color:#1f2328")
                for c in row.index]

    return df.style.apply(colour_row, axis=1)
