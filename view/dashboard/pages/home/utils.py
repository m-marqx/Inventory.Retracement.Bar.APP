import dash_bootstrap_components as dbc


class DropdownMenuItems:
    """A class representing dropdown menu items for a Dash application.

    Parameters
    ----------
    lang : dict
        A dictionary containing language translations.

    Attributes
    ----------
    lang : dict
        A dictionary containing language translations.

    """

    def __init__(self, lang):
        self.lang = lang

    @property
    def ema_ohlc_items(self):
        """Dropdown menu items for EMA OHLC selection.

        Returns
        -------
        list
            A list of dbc.DropdownMenuItem objects representing the EMA
            OHLC items.

        """
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="ema_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="ema_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="ema_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="ema_close"),
        ]

    @property
    def macd_ohlc_items(self):
        """Dropdown menu items for MACD OHLC selection.

        Returns
        -------
        list
            A list of dbc.DropdownMenuItem objects representing the MACD
            OHLC items.

        """
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="macd_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="macd_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="macd_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="macd_close"),
        ]

    @property
    def cci_ohlc_items(self):
        """Dropdown menu items for CCI OHLC selection.

        Returns
        -------
        list
            A list of dbc.DropdownMenuItem objects representing the CCI
            OHLC items.

        """
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="cci_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="cci_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="cci_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="cci_close"),
        ]

    @property
    def source_ohlc_items(self):
        """Dropdown menu items for source OHLC selection.

        Returns
        -------
        list
            A list of dbc.DropdownMenuItem objects representing the
            source OHLC items.

        """
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="source_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="source_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="source_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="source_close"),
        ]

    @property
    def cci_ma_type_items(self):
        """Dropdown menu items for CCI moving average type selection.

        Returns
        -------
        list
            A list of dbc.DropdownMenuItem objects representing the CCI
            moving average type items.

        """
        return [
            dbc.DropdownMenuItem("SMA", id="sma"),
            dbc.DropdownMenuItem("EMA", id="ema"),
        ]

    @property
    def indicators_filter(self):
        """Filter options for indicators.

        Returns
        -------
        list
            A list of dictionaries representing the indicator filter
            options.

        """
        return [
            {"label": "EMA", "value": "ema"},
            {"label": "CCI", "value": "cci"},
            {"label": "MACD", "value": "macd"},
        ]
