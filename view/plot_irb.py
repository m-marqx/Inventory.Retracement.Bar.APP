# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
from model.utils import DataProcess

class Plot:
    def __init__(self, dataframe):
        self.data_frame = dataframe.copy()
        if "open_time" in self.data_frame.columns:
            if "open_time" == self.data_frame.index.dtype == "<M8[ns]":
                self.data_frame["Date"] = self.data_frame.index
            else:
                self.data_frame["Date"] = pd.to_datetime(self.data_frame["open_time"], unit="ms")

        elif "Date" not in self.data_frame.columns:
            self.data_frame["Date"] = self.data_frame.index

        self.data_frame["Exit_Price"] = DataProcess(self.data_frame).exit_price()
        self.fig = None
        self.fig2 = None

    def winrate(self):
        if "Win_Rate" not in self.data_frame.columns:
            self.data_frame["Win_Rate"] = ((self.data_frame["Result"] > 0).cumsum()) / (
                (self.data_frame["Result"] < 0).cumsum()
            )

        self.fig = px.histogram(
            (
                self.data_frame.query("(`Win_Rate` > 0) and (`Close_Position` == True)")
            ).iloc[:, -1],
            histnorm="probability",
        )
        self.fig
        return self

    def results(self):
        self.fig = px.histogram(
            self.data_frame.query("`Close_Position` == True")["Result"]
        )
        return self

    def chart(self):
        # Criar um gráfico de candlesticks com Plotly Express
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

        # Adicionar linhas para ema, Entry_Price, Take_Profit e Stop_Loss
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

        # adicionando o range slider
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
                                count=1, label="1 dia", step="day", stepmode="backward"
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
                                count=1, label="1 ano", step="year", stepmode="backward"
                            ),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=False),
            ),
        )

        # adicionando botões de zoom personalizados
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
                                    "xaxis.range[0]": self.fig["layout"]["xaxis"]["range"][0]
                                    * 0.5
                                    if self.fig["layout"]["xaxis"]["range"]
                                    else None,
                                    "xaxis.range[1]": self.fig["layout"]["xaxis"]["range"][1]
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
                                    "xaxis.range[0]": self.fig["layout"]["xaxis"]["range"][0]
                                    * 2
                                    if self.fig["layout"]["xaxis"]["range"]
                                    else None,
                                    "xaxis.range[1]": self.fig["layout"]["xaxis"]["range"][1]
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

        # Exibir o gráfico
        return self

    # %%
    def trading_results(self):
        # Criar um gráfico de candlesticks com Plotly Express
        self.fig1 = go.Figure(
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

        # Adicionar linhas para ema, Entry_Price, Take_Profit e Stop_Loss
        self.fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["ema"],
                name="EMA",
                line=dict(color="white"),
            )
        )
        self.fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Entry_Price"],
                name="Entry Price",
                line=dict(color="yellow"),
            )
        )
        self.fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Take_Profit"],
                name="Take Profit",
                line=dict(color="lime"),
            )
        )
        self.fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Stop_Loss"],
                name="Stop Loss",
                line=dict(color="red"),
            )
        )

        self.fig1.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Exit_Price"],
                name="Exit",
                mode="markers",
                marker=dict(color="yellow"),
            )
        )

        # Criar um segundo gráfico com a coluna "resultado" e "cumulative_result"
        self.fig2 = go.Figure()
        self.fig2.add_trace(
            go.Scatter(
                x=self.data_frame["Date"],
                y=self.data_frame["Cumulative_Result"],
                name="Cumulative Result",
                line=dict(color="white"),
            )
        )

        # Criar uma figura combinando os dois gráficos
        self.fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.2, 0.8],
        )  # especifica a altura da primeira e segunda linha
        self.fig.update_xaxes(rangeslider_visible=False)

        # # Adicionar o subplot na segunda linha e primeira coluna
        self.fig.add_trace(self.fig2.data[0], row=1, col=1)
        for trace in self.fig2.data[1:]:
            self.fig.add_trace(trace, row=1, col=1)

        # Adicionar o gráfico de candlesticks na primeira linha e primeira coluna
        self.fig.add_trace(self.fig1.data[0], row=2, col=1)
        for trace in self.fig1.data[1:]:
            self.fig.add_trace(trace, row=2, col=1)

        # Atualizar o layout da figura
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
            xaxis_rangeslider_visible=False,  # remove o range slider
        )

        # Exibir o gráfico
        return self

    def fig_to_html(self, title: str, open_file: bool = False):
        pio.write_html(self.fig, file=title + ".html", auto_open=open_file)

    def grouped_lines(self, index: bool = False):
        fig = go.Figure()
        if index:
            self.data_frame["date"] = self.data_frame.index
        colors = []

        # Define a lista de cores a serem utilizadas

        color_idx = 0
        for i, col in enumerate(self.data_frame.columns):
            if i % 100 == 0:
                color_idx += 1
            color = "rgb({},{},{})".format(
                (color_idx * 50) % 256,
                (color_idx * 100) % 256,
                (color_idx * 150) % 256,
            )
            colors.append(color)
            if index:
                fig.add_trace(
                    go.Scatter(
                        x=self.data_frame["date"],
                        y=self.data_frame[col],
                        name=col,
                        line=dict(color=color)
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        y=self.data_frame[col],
                        name=col,
                        line=dict(color=color),
                        hovertemplate="(%{x}, %{y})",
                    )
                )

        fig.update_layout(
            title={
                "text": "Hoffman Inventory Retracement Bar",
                "x": 0.5,
            },
            template="plotly_dark",
            font=dict(
                family="Georgia",
                size=18,
            ),
            legend_title="Parametros",
            showlegend=True,
        )

        if index:
            fig.update_layout(
                xaxis=dict(
                    title="Data",
                    type="date",
                    tickformat="%d/%m/%Y",
                ),
            )

        return fig
