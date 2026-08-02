"""National view: aggregate demand, forecast vs actual, KPI tiles.

Aggregate SERVER-SIDE. Never overlay all 90 series - unreadable and slow.
"""

import dash
from dash import html

dash.register_page(__name__, path="/", name="Overview")

layout = html.Div([html.H2("Overview"), html.P("TODO: KPI tiles + national forecast")])
