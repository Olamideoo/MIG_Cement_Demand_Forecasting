"""Inventory and reorder alerts.

The DataTable is what operations actually act on - sorted by days-to-stockout
ascending, CSV export enabled. Severity needs a text/icon cue as well as colour:
this gets read on site tablets in bad light.

Heatmap does the all-sites view better than overlaid lines: 30 rows x 8 weeks
of projected utilisation.
"""

import dash
from dash import html

dash.register_page(__name__, path="/inventory", name="Inventory & Alerts")

layout = html.Div([html.H2("Inventory & Alerts"), html.P("TODO: reorder table + heatmap")])
