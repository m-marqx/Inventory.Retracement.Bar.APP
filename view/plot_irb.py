import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
from model.utils import BrokerEmulator


class Plot:
    def __init__(self, dataframe):
        self.data_frame = dataframe.copy()
        if "open_time" in self.data_frame.columns:
            if "open_time" == self.data_frame.index.dtype == "<M8[ns]":
                self.data_frame["Date"] = self.data_frame.index
            else:
                self.data_frame["Date"] = pd.to_datetime(
                    self.data_frame["open_time"], unit="ms"
                )

        elif "Date" not in self.data_frame.columns:
            self.data_frame["Date"] = self.data_frame.index

        self.data_frame["Exit_Price"] = (
            BrokerEmulator(self.data_frame)
            .exit_price()
        )

        self.fig = None
        self.fig2 = None

    def winrate(self):
        if "Win_Rate" not in self.data_frame.columns:
            self.data_frame["Win_Rate"] = (
                (self.data_frame["Result"] > 0).cumsum()
                / (self.data_frame["Result"] < 0).cumsum()
            )

        self.fig = px.histogram(
            self.data_frame.query(
                "(`Win_Rate` > 0) and (`Close_Position` == True)"
            ).iloc[:, -1],
            histnorm="probability",
        )
        return self

    def results(self):
        self.fig = px.histogram(
            self.data_frame.query("`Close_Position` == True")["Result"]
        )
        return self

    def chart(self):
        self.fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.data_frame["Date"],
                    open=self.data_frame["open"],
                    high=self.data_frame["high"],
                    low=self.data_frame["low"],
                    close=self.data_frame["close"],
                )
            ]
        )

        self.fig.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["ema"],
                name="EMA",
                line=dict(color="white"),
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Entry_Price"],
                name="Entry Price",
                line=dict(color="yellow"),
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Take_Profit"],
                name="Take Profit",
                line=dict(color="lime"),
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Stop_Loss"],
                name="Stop Loss",
                line=dict(color="red"),
            )
        )

        self.fig.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Exit_Price"],
                name="Exit",
                mode="markers",
                marker=dict(color="yellow"),
            )
        )

        self.fig.update_layout(
            title_text="Hoffman Inventory Retracement Bar",
            template="plotly_dark",
            font=dict(
                family="Georgia",
                size=18,
            ),
            legend_title="Trade Signals",
            showlegend=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1,
                                label="1 dia",
                                step="day",
                                stepmode="backward",
                            ),
                            dict(
                                count=7,
                                label="1 semana",
                                step="day",
                                stepmode="backward",
                            ),
                            dict(
                                count=1,
                                label="1 mês",
                                step="month",
                                stepmode="backward",
                            ),
                            dict(
                                count=6,
                                label="6 meses",
                                step="month",
                                stepmode="backward",
                            ),
                            dict(
                                count=1,
                                label="1 ano",
                                step="year",
                                stepmode="backward",
                            ),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=False),
            ),
        )

        self.fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Zoom In",
                            method="relayout",
                            args=[
                                {
                                    "xaxis.range": [None, None],
                                    "yaxis.range": [None, None],
                                    "xaxis.range[0]": (
                                        self.fig["layout"]["xaxis"]["range"][0]
                                    )
                                    * 0.5
                                    if self.fig["layout"]["xaxis"]["range"]
                                    else None,
                                    "xaxis.range[1]": (
                                        self.fig["layout"]["xaxis"]["range"][1]
                                    )
                                    * 0.5
                                    if self.fig["layout"]["xaxis"]["range"]
                                    else None,
                                }
                            ],
                        ),
                        dict(
                            label="Zoom Out",
                            method="relayout",
                            args=[
                                {
                                    "xaxis.range": [None, None],
                                    "yaxis.range": [None, None],
                                    "xaxis.range[0]": (
                                        self.fig["layout"]["xaxis"]["range"][0]
                                        * 2
                                    )
                                    if self.fig["layout"]["xaxis"]["range"]
                                    else None,
                                    "xaxis.range[1]": (
                                        self.fig["layout"]["xaxis"]["range"][1]
                                    )
                                    * 2
                                    if self.fig["layout"]["xaxis"]["range"]
                                    else None,
                                }
                            ],
                        ),
                    ],
                )
            ]
        )

        return self

    def trading_results(self):
        fig1 = go.Figure(
            data=[
                go.Candlestick(
                    x=self.data_frame["Date"],
                    open=self.data_frame["open"],
                    high=self.data_frame["high"],
                    low=self.data_frame["low"],
                    close=self.data_frame["close"],
                )
            ]
        )

        fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["ema"],
                name="EMA",
                line=dict(color="white"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Entry_Price"],
                name="Entry Price",
                line=dict(color="yellow"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Take_Profit"],
                name="Take Profit",
                line=dict(color="lime"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Stop_Loss"],
                name="Stop Loss",
                line=dict(color="red"),
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Exit_Price"],
                name="Exit",
                mode="markers",
                marker=dict(color="yellow"),
            )
        )

        self.fig2 = go.Figure()
        self.fig2.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Cumulative_Result"],
                name="Cumulative Result",
                line=dict(color="white"),
            )
        )

        self.fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.2, 0.8],
        )
        self.fig.update_xaxes(rangeslider_visible=False)

        self.fig.add_trace(self.fig2.data[0], row=1, col=1)
        for trace in self.fig2.data[1:]:
            self.fig.add_trace(trace, row=1, col=1)

        self.fig.add_trace(fig1.data[0], row=2, col=1)
        for trace in fig1.data[1:]:
            self.fig.add_trace(trace, row=2, col=1)

        self.fig.update_layout(
            title={
                "text": "Hoffman Inventory Retracement Bar",
                "x": 0.5,
            },
            template="plotly_dark",
            font=dict(
                family="Georgia",
                size=18,
            ),
            legend_title="Trade Signals",
            showlegend=True,
            xaxis_rangeslider_visible=False,
        )

        return self

    def fig_to_html(self, title: str, open_file: bool = False):
        if not os.path.exists("data/html"):
            os.makedirs("data/html")
        file_path = os.path.join("data/html", title + ".html")
        pio.write_html(self.fig, file=file_path, auto_open=open_file)
        return self
