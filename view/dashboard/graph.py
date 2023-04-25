import pandas as pd
import plotly.express as px


class GraphLayout:
    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame
        self.tranp_color = "rgba(0,0,0,0)"
        self.title_color = "rgba(255,255,255,0.85)"
        self.label_color = "rgba(255,255,255,0.65)"
        self.primary_color = "#8bbb11"
        self.grid_color = "#595959"


    def fig_layout(self, fig, column, symbol: str, interval: str):
        ticks = self.data_frame[column].std() / 2
        fig.update_layout(
            paper_bgcolor=self.tranp_color,
            plot_bgcolor=self.tranp_color,
            title={
                "text": f"{symbol[:3]}/{symbol[-3:]} - {interval}",
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
                    "text": f"{symbol[-3:]}",
                    "font": {"color": self.label_color},
                },
                color=self.title_color
            ),
        )
        return fig

    def plot_cumulative_results(self, symbol: str, interval: str):
        column = "Cumulative_Result"
        fig = px.line(x=self.data_frame.index, y=self.data_frame[column], color_discrete_sequence=[self.primary_color])
        self.fig_layout(fig, column, symbol, interval)
        return fig

    def plot_close(self, symbol: str, interval: str):
        column = "close"
        fig = px.line(x=self.data_frame.index, y=self.data_frame[column], color_discrete_sequence=[self.primary_color])
        self.fig_layout(fig, column, symbol, interval)
        return fig
