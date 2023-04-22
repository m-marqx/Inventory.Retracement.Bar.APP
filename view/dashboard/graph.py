import pandas as pd
import plotly.express as px


class GraphLayout:
    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame
        self.bgcolor = "rgba(0,0,0,0)"

    def fig_layout(self, fig, column, symbol: str, interval: str):
        ticks = self.data_frame[column].std() / 4
        fig.update_layout(
            paper_bgcolor=self.bgcolor,
            plot_bgcolor=self.bgcolor,
            title={
                "text": f"{symbol} - {interval}",
                "x": 0.5,
            },
            font=dict(
                # family="Georgia",
                size=18,
            ),
            legend_title="Trade Signals",
            showlegend=True,
            xaxis_rangeslider_visible=False,  # remove o range slider
            xaxis_title="Date",
            yaxis_title="USD",
            xaxis=dict(
                showgrid=False,
            ),
            yaxis=dict(
                # gridcolor="#D7DDE5",
                gridcolor="#C2C9D1",
                zeroline=False,
                showgrid=True,
                gridwidth=1,
                dtick=ticks,
            ),
        )
        return fig

    def plot_cumulative_results(self, symbol: str, interval: str):
        column = "Cumulative_Result"
        fig = px.line(x=self.data_frame.index, y=self.data_frame[column])
        self.fig_layout(fig, column, symbol, interval)
        return fig

    def plot_close(self, symbol: str, interval: str):
        column = "close"
        fig = px.line(x=self.data_frame.index, y=self.data_frame[column])
        self.fig_layout(fig, column, symbol, interval)
        return fig
