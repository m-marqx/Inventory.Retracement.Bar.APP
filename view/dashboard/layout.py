import pathlib
import dash
from dash import dcc, DiskcacheManager
import diskcache
import dash_bootstrap_components as dbc

from view.dashboard.pages.general.components import navbar_components

theme = dbc.themes.MORPH
style_sheet = ["assets/style"]
icons = "https://use.fontawesome.com/releases/v5.15.3/css/all.css"
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

page_folder = pathlib.Path("view", "dashboard", "pages")

app = dash.Dash(
    "Dashboard",
    suppress_callback_exceptions=True,
    external_stylesheets=[icons, theme, style_sheet],
    title="Inventory Retracement Bar",
    use_pages=True,
    pages_folder=page_folder,
    background_callback_manager=background_callback_manager,
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dbc.Row(
            [
                dbc.Col(
                    [
                        navbar_components,
                    ],
                    width=True,
                    style={
                        "textAlign": "center",
                        "background-color": "#262626",
                        "margin-bottom": "16px",
                    },
                ),
            ],
            align="end",
        ),
        dbc.Col(id="page-content"),
        dash.page_container,
    ],
    fluid=True,
    style={"font-family": "Open Sans"},
)
