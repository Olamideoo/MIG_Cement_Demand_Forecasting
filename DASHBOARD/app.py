"""Plotly Dash entrypoint. Run: python DASHBOARD/app.py

Calls the FastAPI service for all predictions - no model loading here.
"""

from __future__ import annotations

import dash
from dash import Dash, html

from mig_cement.config import settings

app = Dash(__name__, use_pages=True, pages_folder="pages",
           suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1("MIG Cement Demand Forecasting"),
    html.Nav([
        dash.dcc.Link(p["name"], href=p["relative_path"], style={"marginRight": "1rem"})
        for p in dash.page_registry.values()
    ]),
    html.Hr(),
    dash.page_container,
])

if __name__ == "__main__":
    app.run(debug=True, port=settings.dashboard_port)
