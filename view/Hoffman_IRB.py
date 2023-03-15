#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
#%%

def get_date_column(dataframe):
    if "open_time" in dataframe.columns:
        dataframe["Date"] = pd.to_datetime(dataframe["open_time"], unit="ms")
    elif "Date" not in dataframe.columns:
        dataframe["Date"] = dataframe.index
        return dataframe["Date"]

def IRB_plot(dataframe):
    data_frame = dataframe.copy()

    if "Win Rate" not in data_frame.columns:
        data_frame["Win Rate"] = ((data_frame["Result"] > 0).cumsum()) / ((data_frame["Result"] < 0).cumsum())

    fig = px.histogram(
        (data_frame.query("(`Win Rate` > 0) and (`Close Position` == True)")).iloc[:, -1],
        histnorm="probability",
    )
    fig2 = px.histogram(data_frame.query("`Close Position` == True").iloc[:, -6])
    fig.show()
    fig2.show()

    data_frame.plot(x="date", y="Cumulative_Result", kind="line")

    # Criar um gráfico de candlesticks com Plotly Express
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data_frame["date"],
                open=data_frame["open"],
                high=data_frame["high"],
                low=data_frame["low"],
                close=data_frame["close"],
            )
        ]
    )

    # Adicionar linhas para ema, Entry_Price, Take_Profit e Stop_Loss
    fig.add_trace(
        go.Scatter(x=data_frame["date"], y=data_frame["ema"], name="EMA", line=dict(color="white"))
    )
    fig.add_trace(
        go.Scatter(
            x=data_frame["date"],
            y=data_frame["Entry_Price"],
            name="Entry Price",
            line=dict(color="yellow"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_frame["date"],
            y=data_frame["Take_Profit"],
            name="Take Profit",
            line=dict(color="lime"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_frame["date"], y=data_frame["Stop_Loss"], name="Stop Loss", line=dict(color="red")
        )
    )

    # adicionando o range slider
    fig.update_layout(
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
                buttons=list([
                    dict(count=1, label="1 dia", step="day", stepmode="backward"),
                    dict(count=7, label="1 semana", step="day", stepmode="backward"),
                    dict(count=1, label="1 mês", step="month", stepmode="backward"),
                    dict(count=6, label="6 meses", step="month", stepmode="backward"),
                    dict(count=1, label="1 ano", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=False
            ),
            type="date"
        )
    )

# adicionando botões de zoom personalizados
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Zoom In',
                        method='relayout',
                        args=[{'xaxis.range': [None, None], 'yaxis.range': [None, None],
                            'xaxis.range[0]': fig['layout']['xaxis']['range'][0]*0.5 if fig['layout']['xaxis']['range'] else None,
                            'xaxis.range[1]': fig['layout']['xaxis']['range'][1]*0.5 if fig['layout']['xaxis']['range'] else None}]
                    ),
                    dict(
                        label='Zoom Out',
                        method='relayout',
                        args=[{'xaxis.range': [None, None], 'yaxis.range': [None, None],
                            'xaxis.range[0]': fig['layout']['xaxis']['range'][0]*2 if fig['layout']['xaxis']['range'] else None,
                            'xaxis.range[1]': fig['layout']['xaxis']['range'][1]*2 if fig['layout']['xaxis']['range'] else None}]
                    )
                ]
            )
        ]
    )

    # Exibir o gráfico
    fig.show()

# %%
def show_trading_results(dataframe):
    data_frame = dataframe.copy()

    if "Win Rate" in data_frame.columns:
        data_frame["Win Rate"] = data_frame["Win Rate"]
    else:
        wins = (data_frame["Result"] > 0).cumsum()
        losses = (data_frame["Result"] < 0).cumsum()
        win_rate = wins / (wins + losses)
        data_frame["Win Rate"] = win_rate


    fig = px.histogram(
        (data_frame.query("(`Win Rate` > 0) and (`Close Position` == True)")).iloc[:, -1],
        histnorm="probability",
    )
    fig2 = px.histogram(data_frame.query("`Close Position` == True").iloc[:, -6])
    fig.show()
    fig2.show()

    data_frame.plot(x="date", y="Cumulative_Result", kind="line")

    # Criar um gráfico de candlesticks com Plotly Express
    fig1 = go.Figure(
        data=[
            go.Candlestick(
                x=data_frame["date"],
                open=data_frame["open"],
                high=data_frame["high"],
                low=data_frame["low"],
                close=data_frame["close"],
            )
        ]
    )

    # Adicionar linhas para ema, Entry_Price, Take_Profit e Stop_Loss
    fig1.add_trace(
        go.Scatter(x=data_frame["date"], y=data_frame["ema"], name="EMA", line=dict(color="white"))
    )
    fig1.add_trace(
        go.Scatter(
            x=data_frame["date"],
            y=data_frame["Entry_Price"],
            name="Entry Price",
            line=dict(color="yellow"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=data_frame["date"],
            y=data_frame["Take_Profit"],
            name="Take Profit",
            line=dict(color="lime"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=data_frame["date"], y=data_frame["Stop_Loss"], name="Stop Loss", line=dict(color="red")
        )
    )

    # Criar um segundo gráfico com a coluna "resultado" e "cumulative_result"
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=data_frame["date"],
            y=data_frame["Cumulative_Result"],
            name="Cumulative Result",
            line=dict(color="white"),
        )
    )

    # Criar uma figura combinando os dois gráficos
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.2, 0.8],
    )  # especifica a altura da primeira e segunda linha
    # row_widths=[1]) # especifica a largura de cada coluna

    # # Adicionar o subplot na segunda linha e primeira coluna
    fig.add_trace(fig3.data[0], row=1, col=1)
    for trace in fig3.data[1:]:
        fig.add_trace(trace, row=1, col=1)

    # Adicionar o gráfico de candlesticks na primeira linha e primeira coluna
    fig.add_trace(fig1.data[0], row=2, col=1)
    for trace in fig1.data[1:]:
        fig.add_trace(trace, row=2, col=1)

    # Atualizar o layout da figura
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
        legend_title="Trade Signals",
        showlegend=True,
        xaxis_rangeslider_visible=False,  # remove o range slider
    )

    # Exibir o gráfico
    fig.show()