from typing_extensions import Literal

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from klib import corr_mat
from klib.utils import _validate_input_bool, _validate_input_range

from screeninfo import get_monitors

def corr_interactive_plot(
    data: pd.DataFrame,
        split: Literal["pos", "neg", "high", "low"] | None = None,
    threshold: float = 0,
    target: pd.Series | str | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    cmap: str = "BrBG",
    figsize: tuple[float, float] = (12, 10),
    annot: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Create a two-dimensional visualization of the correlation between
    feature-columns using Plotly's Heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into a Pandas DataFrame. If a
        Pandas DataFrame is provided, the index/column information is
        used to label the plots.
    split : Optional[str], optional
        Type of split to be performed
        {None, "pos", "neg", "high", "low"}, by default None

        - None: visualize all correlations between the feature-columns

        - pos: visualize all positive correlations between the
            feature-columns above the threshold

        - neg: visualize all negative correlations between the
            feature-columns below the threshold

        - high: visualize all correlations between the
            feature-columns for which abs(corr) > threshold is True

        - low: visualize all correlations between the
            feature-columns for which abs(corr) < threshold is True

    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold,
        by default 0 unless split = "high" or split = "low", in
        which case the default is 0.3

    target : Optional[pd.Series | str], optional
        Specify a target for correlation. For example, the label column
        to generate only the correlations between each feature and the
        label, by default None

    method : Literal['pearson', 'spearman', 'kendall'], optional
        Method for correlation calculation:
        {"pearson", "spearman", "kendall"}, by default "pearson"

        - pearson: measures linear relationships and requires normally
            distributed and homoscedastic data.
        - spearman: ranked/ordinal correlation, measures monotonic
            relationships.
        - kendall: ranked/ordinal correlation, measures monotonic
            relationships. Computationally more expensive but more
            robust in smaller datasets than "spearman".

    cmap : str, optional
        The mapping from data values to color space, plotly
        colormap name or object, or list of colors, by default "BrBG"

    figsize : tuple[float, float], optional
        Use to control the figure size, by default (12, 10)

    annot : bool, optional
        Use to show or hide annotations, by default True

    **kwargs : optional
        Additional elements to control the visualization of the plot.
            These additional arguments will be passed to the `go.Heatmap`
            function in Plotly.

        Specific kwargs used in this function:

        - colorscale: str or list, optional
            The colorscale to be used for the heatmap. It controls the
            mapping of data values to colors in the heatmap.

        - zmax: float, optional
            The maximum value of the color scale. It limits the upper
            range of the colorbar displayed on the heatmap.

        - zmin: float, optional
            The minimum value of the color scale. It limits the lower
            range of the colorbar displayed on the heatmap.

        - text: pd.DataFrame, optional
            A DataFrame containing text to display on the heatmap. This
            text will be shown on the heatmap cells corresponding to the
            correlation values.

        - texttemplate: str, optional
            A text template string to format the text display on the
            heatmap. This allows you to customize how the text appears,
            including the display of the correlation values.

        - textfont: dict, optional
            A dictionary specifying the font properties for the text on
            the heatmap. You can customize the font size, color, family,
            etc., for the text annotations.

        - x: list, optional
            The list of column names for the x-axis of the heatmap. It
            allows you to customize the labels displayed on the x-axis.

        - y: list, optional
            The list of row names for the y-axis of the heatmap. It
            allows you to customize the labels displayed on the y-axis.

        - z: pd.DataFrame, optional
            The 2D array representing the correlation matrix to be
            visualized. This is the core data for generating the heatmap,
            containing the correlation values.

        - Many more kwargs are available, e.g., "hovertemplate" to control
            the legend hover template, or options to adjust the borderwidth
            and opacity of the heatmap. For a comprehensive list of
            available kwargs, please refer to the Plotly Heatmap documentation.

        Kwargs can be supplied through a dictionary of key-value pairs
        (see above) and can be found in Plotly Heatmap documentation.

    Returns
    -------
    heatmap : plotly.graph_objs._figure.Figure
        A Plotly Figure object representing the heatmap visualization of
        feature correlations.
    """
    # Validate Inputs
    _validate_input_range(threshold, "threshold", -1, 1)
    _validate_input_bool(annot, "annot")

    data = pd.DataFrame(data).iloc[:, ::-1]

    corr = corr_mat(
        data,
        split=split,
        threshold=threshold,
        target=target,
        method=method,
        colored=False,
    )

    mask = np.zeros_like(corr, dtype=bool)

    if target is None:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        np.fill_diagonal(corr.to_numpy(), np.nan)
        corr = corr.where(mask == 1)
    else:
        corr = corr.iloc[::-1,:]

    vmax = np.round(np.nanmax(corr) - 0.05, 2)
    vmin = np.round(np.nanmin(corr) + 0.05, 2)

    if annot:
        vtext = corr.round(2).fillna("")
    else:
        vtext = None

    # Specify kwargs for the heatmap
    kwargs = {
        "colorscale": cmap,
        "zmax": vmax,
        "zmin": vmin,
        "text": vtext,
        "texttemplate": "%{text}",
        "textfont": {"size": 12},
        "x": corr.columns,
        "y": corr.index,
        "z": corr,
        **kwargs,
    }

    # Draw heatmap with masked corr and default settings
    heatmap = go.Figure(
        data=go.Heatmap(
            hoverongaps=False,
            xgap=1,
            ygap=1,
            **kwargs,
        )
    )

    for monitor in get_monitors():
        if monitor.is_primary:
            dpi = monitor.width / (monitor.width_mm / 25.4)

    if dpi is None:
        try:
            monitor = get_monitors()[0]
            dpi = monitor.width / (monitor.width_mm / 25.4)
        except Exception as exc:
            raise LookupError("Monitor doesn't exist") from exc

    heatmap.update_layout(
        title=f"Feature-correlation ({method})",
        title_font={"size":24},
        title_x=0.5,
        autosize=True,
        width=figsize[0] * dpi,
        height=(figsize[1] + 1) * dpi,
        xaxis={"autorange": "reversed"},
    )

    return heatmap
