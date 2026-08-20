"""MIG Cement Demand Forecasting — operations dashboard.

    streamlit run DASHBOARD/app.py

Loads the trained model and runs the inventory simulation locally. All data access
goes through `data.py`, so moving to the FastAPI service later means reimplementing
that module and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))       # let views import `data`

st.set_page_config(page_title="MIG Cement Demand Forecasting",
                   page_icon="🏗️", layout="wide")

from views import baseline, inventory, overview, performance, site_detail  # noqa: E402

import data as dat  # noqa: E402

PAGES = {
    "Overview": overview.render,
    "Before: the baseline": baseline.render,
    "Site detail": site_detail.render,
    "Inventory & alerts": inventory.render,
    "Model performance": performance.render,
}


def main() -> None:
    st.sidebar.title("MIG Cement")
    st.sidebar.caption("8-week demand forecasting and reorder planning")

    ok, message = dat.artefacts_present()
    if not ok:
        st.title("MIG Cement Demand Forecasting")
        st.error(message)
        st.stop()

    page = st.sidebar.radio("View", list(PAGES), label_visibility="collapsed")

    meta = dat.get_model_metadata()
    st.sidebar.divider()
    st.sidebar.caption(
        f"**Model** {meta.get('name', 'n/a')}  \n"
        f"{meta.get('grain', '')}  \n"
        f"horizon {meta.get('horizon_weeks', '?')} weeks  \n"
        f"trained to {meta.get('trained_on', {}).get('to', '?')}")
    if st.sidebar.button("Reload model and data",
                         help="Re-reads the database, the saved model and the "
                              "forecasts, then re-runs the simulation. Use after "
                              "`make pipeline`."):
        # Both caches, not just cache_data. The model is held in cache_resource,
        # so clearing only cache_data would refresh the metadata in this sidebar
        # while the old model object stayed in memory - new numbers, stale
        # predictions, and nothing on screen to say so.
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.title("MIG Cement Demand Forecasting")
    PAGES[page]()


if __name__ == "__main__":
    main()
