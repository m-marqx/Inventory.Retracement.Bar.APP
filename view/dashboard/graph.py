import pandas as pd
import plotly.express as px
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

        #In the pair names, the "mark_price" has the same values as the "coin_margined".
        if self.api == "mark_price":
            self.api = "coin_margined"

        kline_api = KlineAPI(self.symbol,self.interval,self.api)
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
                    "text":"Date",
                    "font": {"color": self.label_color},
                },
                color=self.title_color
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
                color=self.title_color
            ),
        )
        return fig

    def plot_cumulative_results(self):
        column = "Cumulative_Result"
        fig = px.line(x=self.data_frame.index, y=self.data_frame[column], color_discrete_sequence=[self.primary_color])
        self.fig_layout(fig, column)
        return fig

    def plot_close(self):
        column = "close"
        fig = px.line(x=self.data_frame.index, y=self.data_frame[column], color_discrete_sequence=[self.primary_color])
        self.fig_layout(fig, column)
        return fig
