#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
#%%

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
