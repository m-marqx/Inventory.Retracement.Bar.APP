import dash
from dash import Input, Output, callback

class NavLinkPages:
    @callback(
        Output("home", "active"),
        Output("backtest", "active"),
        Output("ml", "active"),
        Input("home", "n_clicks"),
        Input("backtest", "n_clicks"),
        Input("ml", "n_clicks"),
    )

    def toggle_active_links(home, backtest, ml):
        ctx = dash.callback_context

        pages = {
            "home": True,
            "backtest": False,
            "ml": False,
        }

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        for i in pages:
            if i == button_id:
                pages[i] = True
            else:
                pages[i] = False

        pages_values = list(pages.values())

        return pages_values[0], pages_values[1], pages_values[2]
