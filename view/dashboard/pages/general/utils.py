import io
import base64
import pandas as pd

import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_ag_grid as dag

def intervals(lang):
    return [
        {"label": "1min", "value": "1m"},
        {"label": "5min", "value": "5m"},
        {"label": "15min", "value": "15m"},
        {"label": "30min", "value": "30m"},
        {"label": "1h", "value": "1h"},
        {"label": "2h", "value": "2h"},
        {"label": "4h", "value": "4h"},
        {"label": "6h", "value": "6h"},
        {"label": "8h", "value": "8h"},
        {"label": "12h", "value": "12h"},
        {"label": "1d", "value": "1d"},
        {"label": "3d", "value": "3d"},
        {"label": "1w", "value": "1w"},
        {"label": "1M", "value": "1M"},
        {"label": lang["CUSTOM"], "value": "Custom"},
    ]


def api_types(lang):
    return [
        {"label": lang["SPOT"], "value": "spot"},
        {"label": lang["FUTURES"], "value": "coin_margined"},
        {"label": lang["MARK_PRICE"], "value": "mark_price"},
        {"label": lang["CUSTOM"], "value": "custom"},
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
        The label used to retrieve the translated name from the "lang"
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
        id_prefix: str,
        is_open: bool = False,
    ):
        """
        Initialize a MenuCollapse instance.

        Parameters
        ----------
        lang : dict
            A dictionary containing language translations.
        label : str
            The label used to retrieve the translated name from the
            "lang" dictionary.
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
        self.is_open = is_open

        self.button = dbc.Button(
            [
                self.label_name,
                html.I(
                    className="fa fa-chevron-down ml-2",
                    id=f"{self.id_prefix}_icon",
                    style={"transformY": "2px"},
                ),
            ],
            id=f"{id_prefix}_button",
            className="d-grid gap-2 col-6 mx-auto w-100",
            outline=True,
            color="secondary",
        )

    @property
    def simple_collapse(self):
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
            is_open=self.is_open,
        )

        return dbc.Col([self.button, collapse])

    def collapse_with_inside_collapse(self, inside_component):
        """
        Create a collapsible menu item.

        Returns
        -------
        dbc.Col
            A column component containing the collapse and button
            components.
        """
        collapse = dbc.Collapse(
            dbc.Card(
                [
                    dbc.CardBody(
                        self.component,
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            inside_component,
                        ),
                    ),
                ]
            ),
            id=f"{self.id_prefix}_collapse",
            is_open=self.is_open,
        )

        return dbc.Col([self.button, collapse])

def upload_component(label: str, id_prefix: str, button_class: str) -> dbc.Col:
    """
    Generate a Dash Bootstrap Component column containing an upload
    button.

    Parameters:
    -----------
    label : str
        The label for the upload button.
    id_prefix : str
        Prefix to be added to the IDs of the button and upload
        components.
    button_class : str
        The class name for styling the button.

    Returns:
    --------
    dbc.Col
        A Dash Bootstrap Component column containing the upload button.
    """
    return dbc.Col(
        dbc.Col(
            [
                dbc.Button(
                    color="primary",
                    id=f"{id_prefix}_button",
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

def table_component(
    data_frame: pd.DataFrame,
    id_prefix: str,
    class_name: str = "ag-theme-alpine-dark",
    use_pagination: bool = False
) -> dag.AgGrid:
    """
    Generate an Ag-Grid table component for displaying a DataFrame.

    Parameters:
    -----------
    data_frame : pd.DataFrame
        The DataFrame to be displayed in the table.
    id_prefix : str
        Prefix to be added to the IDs of the table components.
    class_name : str, optional
        The class name for styling the table,
        by default "ag-theme-alpine-dark".
    use_pagination : bool, optional
        Whether to enable pagination in the table, by default False.

    Returns:
    --------
    dag.AgGrid
        An Ag-Grid table component for displaying the DataFrame.
    """
    if use_pagination:
        pagination_dict = {"pagination": True, "paginationPageSize":10}
    else:
        pagination_dict = {"pagination": False}

    return dag.AgGrid(
        id=f"{id_prefix}-table",
        rowData=data_frame.to_dict("records"),
        columnDefs=[
            {"headerName": col, "field": col}
            for col in data_frame.columns
        ],
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        columnSize="responsiveSizeToFit",
        dashGridOptions=pagination_dict,
        className=class_name
    )

def content_parser(contents, filename):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if "csv" in filename:
        file_name = io.StringIO(decoded.decode("utf-8"))
        data_frame = pd.read_csv(file_name)

    elif "parquet" in filename:
        file_name = io.BytesIO(decoded)
        data_frame = pd.read_parquet(file_name)
    else:
        return None
    return data_frame
