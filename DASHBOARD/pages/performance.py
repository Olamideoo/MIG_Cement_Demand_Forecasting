"""Model performance and drift.

Forecast-vs-actual MUST shade periods where consumption was stock-constrained
(39.7% of historical rows). Without that, the page shows the model over-forecasting
when the model was right and the site simply ran out.
"""

import dash
from dash import html

dash.register_page(__name__, path="/performance", name="Model Performance")

layout = html.Div([html.H2("Model Performance"), html.P("TODO: accuracy by site/horizon")])
