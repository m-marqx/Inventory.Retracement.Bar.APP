import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from controller.api.klines_api import KlineAPI


class GraphLayout:
    def __init__(
        self,
        data_frame: pd.DataFrame,
        symbol: str,
        interval: str,
        api: str,
    ):
        self.data_frame = data_frame
        self.tranp_color = "rgba(0,0,0,0)"
        self.title_color = "rgba(255,255,255,0.85)"
        self.label_color = "rgba(255,255,255,0.65)"
        self.primary_color = "#8bbb11"
        self.grid_color = "#595959"
        self.symbol = symbol
        self.interval = interval
        self.api = api

    def fig_layout(self, fig, column):
        ticks = self.data_frame[column].std() / 2

        # In the pair names, the "mark_price" has the same values as the "coin_margined".
        if self.api == "mark_price":
            self.api = "coin_margined"

        kline_api = KlineAPI(self.symbol, self.interval, self.api)
        symbol_info = kline_api.get_exchange_symbol_info()
        coin_name = list(symbol_info.baseAsset)[0]
        currency_name = list(symbol_info.quoteAsset)[0]
        pair_name = f"{coin_name}/{currency_name}"

        fig.update_layout(
            paper_bgcolor=self.tranp_color,
            plot_bgcolor=self.tranp_color,
            title={
                "text": f"{pair_name} - {self.interval}",
                "x": 0.5,
                "font": {"color": self.title_color},
            },
            font=dict(
                size=18,
            ),
            legend_title="Trade Signals",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                showgrid=False,
                title={
                    "text": "Date",
                    "font": {"color": self.label_color},
                },
                color=self.title_color,
            ),
            yaxis=dict(
                zeroline=False,
                showgrid=True,
                gridwidth=1,
                griddash="dash",
                gridcolor=self.grid_color,
                exponentformat="none",
                dtick=ticks,
                title={
                    "text": f"{currency_name}",
                    "font": {"color": self.label_color},
                },
                color=self.title_color,
            ),
        )
        return fig

    def custom_fig_layout(self, fig, column):
        ticks = self.data_frame[column].std() / 2

        fig.update_layout(
            paper_bgcolor=self.tranp_color,
            plot_bgcolor=self.tranp_color,
            title={
                "text": "Custom",
                "x": 0.5,
                "font": {"color": self.title_color},
            },
            font=dict(
                size=18,
            ),
            legend_title="Trade Signals",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                showgrid=False,
                color=self.title_color,
            ),
            yaxis=dict(
                zeroline=False,
                showgrid=True,
                gridwidth=1,
                griddash="dash",
                gridcolor=self.grid_color,
                exponentformat="none",
                dtick=ticks,
                color=self.title_color,
            ),
        )
        return fig

    def plot_cumulative_results(self):
        column = "Cumulative_Result"
        fig = px.line(
            x=self.data_frame.index,
            y=self.data_frame[column],
            color_discrete_sequence=[self.primary_color],
        )
        self.fig_layout(fig, column)
        return fig

    def plot_single_linechart(self, column):
        fig = px.line(
            x=self.data_frame.index,
            y=self.data_frame[column],
            color_discrete_sequence=[self.primary_color],
        )
        self.fig_layout(fig, column)
        return fig

    def plot_close(self):
        column = "close"
        fig = px.line(
            x=self.data_frame.index,
            y=self.data_frame[column],
            color_discrete_sequence=[self.primary_color],
        )
        self.fig_layout(fig, column)
        return fig

    def grouped_lines(self):
        fig = go.Figure()
        total_columns = self.data_frame.shape[1]
        columns = self.data_frame.columns
        first_column = self.data_frame.columns[0]

        if total_columns < 5:
            color_variations = 1
        else:
            color_variations = total_columns // 5

        color_idx = 0
        for i, column in enumerate(columns):
            if i % color_variations == 0:
                color_idx += 1
            if color_idx == 1:
                color = "#d89614"
            elif color_idx == 2:
                color = "#1668dc"
            elif color_idx == 3:
                color = "#642ab5"
            elif color_idx == 4:
                color = "#cb2b83"
            else:
                color = "#d32029"

            fig.add_trace(
                go.Scatter(
                    y=self.data_frame[column],
                    name=column,
                    line=dict(color=color),
                    hovertemplate="(%{x}, %{y})",
                )
            )

        ticks = self.data_frame[first_column].std() / 2

        fig.update_layout(
            paper_bgcolor=self.tranp_color,
            plot_bgcolor=self.tranp_color,
            title={
                "text": f"{self.symbol} - {self.interval}",
                "x": 0.5,
                "font": {"color": self.title_color},
            },
            font=dict(
                size=18,
            ),
            legend_title="Trade Signals",
            showlegend=False,
            xaxis_rangeslider_visible=False,
            xaxis=dict(showgrid=False, color=self.title_color),
            yaxis=dict(
                zeroline=False,
                showgrid=True,
                gridwidth=1,
                griddash="dash",
                gridcolor=self.grid_color,
                exponentformat="none",
                dtick=ticks,
                color=self.title_color,
            ),
        )

        # The x-axis is added using this loop because adding the x-axis with
        # the go.Scatter() method would significantly increase the execution time.
        for scatter in fig.data:
            scatter.x = self.data_frame.index

        return fig
