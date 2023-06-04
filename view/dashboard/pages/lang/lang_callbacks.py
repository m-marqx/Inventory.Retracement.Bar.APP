from dash import Output, Input, State, callback


class LangCallbacks:
    @callback(
        Output("lang_selection", "data"),
        Output("home", "href"),
        Output("backtest", "href"),
        Output("en_US_lang", "active"),
        Output("pt_BR_lang", "active"),
        Input("pt_BR_lang", "n_clicks_timestamp"),
        Input("en_US_lang", "n_clicks_timestamp"),
        State("lang_selection", "data"),
    )
    def lang_selection(pt_BR, en_US, lang_selected):
        # This condition will trigger when the user access the app
        if lang_selected == "?lang=pt_BR":
            pt_BR_lang = True
            en_US_lang = False
        else:
            pt_BR_lang = False
            en_US_lang = True

        # This condition will trigger when the user clicks on the language button
        if pt_BR > en_US:
            lang_selection_data = "?lang=pt_BR"
            pt_BR_lang = True
            en_US_lang = False
        else:
            lang_selection_data = "?lang=en_US"
            pt_BR_lang = False
            en_US_lang = True

        home_url = f"/{lang_selection_data}"
        backtest_url = f"/backtest{lang_selection_data}"
        return (
            lang_selection_data,
            home_url,
            backtest_url,
            en_US_lang,
            pt_BR_lang,
        )
