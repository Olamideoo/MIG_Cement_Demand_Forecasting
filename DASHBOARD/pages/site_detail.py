"""Site drill-down. Contains the primary chart of the whole application:

projected silo level over the 8-week horizon, with add_hline for silo capacity
and reorder point, forecast confidence band beneath, and the zero-crossing
marking the projected stockout.

Build this chart in a notebook first, before any layout work.
"""

import dash
from dash import html

dash.register_page(__name__, path="/site", name="Site Detail")

layout = html.Div([html.H2("Site Detail"), html.P("TODO: silo projection chart")])
