from typing import Optional
from typing_extensions import Literal

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from klib import corr_mat
from klib.utils import _validate_input_bool, _validate_input_range

from screeninfo import get_monitors

def corr_plot_interactive(
    data: pd.DataFrame,
    split: Optional[Literal["pos", "neg", "high", "low"]] = None,
    threshold: float = 0,
    target: Optional[pd.Series | str] = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    cmap: str = "BrBG",
    figsize: tuple[float, float] = (12, 10),
    annot: bool = True,
    **kwargs,
) -> go.Figure:
    """Two-dimensional visualization of the correlation between feature-columns
        using Plotly's Heatmap.

    This function generates a heatmap representation of the correlation between
    feature-columns in the provided 2D dataset. It uses the Plotly library to create
    interactive and visually appealing visualizations.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into a Pandas DataFrame. If a Pandas DataFrame
        is provided, the index/column information is used to label the plots.
    split : Optional[str], optional
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default
        None
            * None: visualize all correlations between the feature-columns.
            * pos: visualize all positive correlations between the feature-columns
                above the threshold.
            * neg: visualize all negative correlations between the feature-columns
                below the threshold.
            * high: visualize all correlations between the feature-columns for
                which abs(corr) > threshold is True.
            * low: visualize all correlations between the feature-columns for which
                abs(corr) < threshold is True.
    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0.
        If split is set to "high" or "low", the default threshold is 0.3.
    target : Optional[pd.Series | str], optional
        Specify the target for correlation. For example, a label column to generate
        only the correlations between each feature and the label, by default None.
    method : Literal['pearson', 'spearman', 'kendall'], optional
        Method used for correlation computation, by default "pearson".
            * "pearson": measures linear relationships and requires normally
                distributed and homoscedastic data.
            * "spearman": ranked/ordinal correlation, measures monotonic
                relationships.
            * "kendall": ranked/ordinal correlation, measures monotonic
                relationships. Computationally more expensive but more robust in
                smaller datasets than "spearman".
    cmap : str, optional
        The mapping from data values to the color space, matplotlib colormap name or
        object, or list of colors, by default "BrBG".
    figsize : tuple[float, float], optional
        Used to control the figure size, by default (12, 10).
    annot : bool, optional
        Used to show or hide annotations, by default True.
    **kwargs : optional
        Additional elements to control the visualization of the plot, e.g.:

            * colorbar: dict, optional
                Dictionary containing colorbar properties such as "title" and "titleside".
            * xaxis: dict, optional
                Dictionary containing x-axis properties like "title" and "showgrid".
            * yaxis: dict, optional
                Dictionary containing y-axis properties like "title" and "showgrid".
            * title: str, optional
                The title of the plot.
            * Many more kwargs are available to customize the plot appearance.

    Returns
    -------
    heatmap : plotly.graph_objs._figure.Figure
        A Plotly Figure object representing the heatmap visualization of feature
        correlations.
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
    vtext = corr.round(2).fillna("")

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
        title_font={"size":18},
        autosize=True,
        width=figsize[0] * dpi,
        height=(figsize[1] + 1) * dpi,
        xaxis={"autorange": "reversed"},
    )

    return heatmap
