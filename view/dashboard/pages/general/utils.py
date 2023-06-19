import dash_bootstrap_components as dbc
from dash import html, dcc

intervals = [
    dbc.DropdownMenuItem("1min", id="1m"),
    dbc.DropdownMenuItem("5min", id="5m"),
    dbc.DropdownMenuItem("15min", id="15m"),
    dbc.DropdownMenuItem("30min", id="30m"),
    dbc.DropdownMenuItem("1h", id="1h"),
    dbc.DropdownMenuItem("2h", id="2h"),
    dbc.DropdownMenuItem("4h", id="4h"),
    dbc.DropdownMenuItem("6h", id="6h"),
    dbc.DropdownMenuItem("8h", id="8h"),
    dbc.DropdownMenuItem("12h", id="12h"),
    dbc.DropdownMenuItem("1d", id="1d"),
    dbc.DropdownMenuItem("3d", id="3d"),
    dbc.DropdownMenuItem("1w", id="1w"),
    dbc.DropdownMenuItem("1M", id="1M"),
]


def api_types(lang):
    return [
        {"label": lang["SPOT"], "value": "spot"},
        {"label": lang["FUTURES"], "value": "coin_margined"},
        {"label": lang["MARK_PRICE"], "value": "mark_price"},
    ]


def result_types(lang):
    return [
        {"label": lang["FIXED"], "value": "Fixed"},
        {"label": lang["NORMAL"], "value": "Normal"},
    ]

class MenuCollapse:
    """
    A class representing a collapsible menu item.

    Parameters
    ----------
    lang : dict
        A dictionary containing language translations.
    label : str
        The label used to retrieve the translated name from the 'lang'
        dictionary.
    component : dbc._components.Row
        The component to be displayed inside the collapsible menu item.
    id_prefix : str
        A prefix used to generate unique IDs for the collapse and button
        components.

    Attributes
    ----------
    label_name : str
        The translated label name.
    component : dbc._components.Row
        The component to be displayed inside the collapsible menu item.
    id_prefix : str
        A prefix used to generate unique IDs for the collapse and button
        components.

    Methods
    -------
    menu_collapse()
        Create a collapsible menu item.

        Returns
        -------
        tuple
            A tuple containing the collapse and button components.
    """

    def __init__(
        self,
        lang: dict,
        label: str,
        component,
        id_prefix: str
    ):
        """
        Initialize a MenuCollapse instance.

        Parameters
        ----------
        lang : dict
            A dictionary containing language translations.
        label : str
            The label used to retrieve the translated name from the
            'lang' dictionary.
        component : dbc._components.Row
            The component to be displayed inside the collapsible menu
            item.
        id_prefix : str
            A prefix used to generate unique IDs for the collapse and
            button components.
        """
        self.label_name = lang[label]
        self.component = component
        self.id_prefix = id_prefix

    @property
    def components(self):
        """
        Create a collapsible menu item.

        Returns
        -------
        tuple
            A tuple containing the collapse and button components.
        """
        collapse = dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    self.component,
                )
            ),
            id=f"{self.id_prefix}_collapse",
            is_open=False,
        )

        button = dbc.Button(
            [
                self.label_name,
                html.I(
                    className="fa fa-chevron-down ml-2",
                    id=f"{self.id_prefix}_icon",
                    style={"transformY": "2px"},
                ),
            ],
            id=f"{self.id_prefix}_button",
            className="d-grid gap-2 col-6 mx-auto w-100",
            outline=True,
            color="secondary",
        )

        return collapse, button

def upload_component(label: str, id_prefix: str, button_class: str):
    return dbc.Col(
        dbc.Col(
            [
                dbc.Button(
                    color="primary",
                    id=f"{id_prefix}-button",
                    class_name=button_class,
                    style={"margin-top": "30px"},
                    children=dbc.Col(
                        dcc.Upload(
                            label,
                            id=f"{id_prefix}-data",
                            multiple=False,
                            style={"opacity": "1"},
                            style_active={"opacity": "0.5"},
                        ),
                    ),
                ),
            ]
        ),
    )

