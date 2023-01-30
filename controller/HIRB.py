#%%
import yfinance as yf
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
#from plotly.subplots import make_subplots
#from datetime import datetime

#%% 

df = yf.download(tickers='EURUSD=X', period='59d', interval='15m')
df.reset_index(drop=True, inplace=True)

#%%
df["EMA"] = ta.ema(df.Close, length=20)
backrollingN = 20
df['slopeEMA'] = df['EMA'].diff(periods=1)
df['slopeEMA'] = df['slopeEMA'].rolling(window=backrollingN).mean()

#%%
TotSignal = [0] * len(df)
slopelimit = 5e-5
percentlimit = 0.45
previous_signal = None

#%%

for row in range(0, len(df)):
    if df.slopeEMA[row] < -slopelimit and (min(df.Open[row], df.Close[row])-df.Low[row])/(df.High[row]-df.Low[row]) > percentlimit:
        current_signal = 1
    elif df.slopeEMA[row] > slopelimit and (df.High[row]-max(df.Open[row], df.Close[row]))/(df.High[row]-df.Low[row]) > percentlimit:
        current_signal = 2
    else:
        current_signal = 0
    if current_signal != previous_signal:
        TotSignal[row] = current_signal
    previous_signal = current_signal

df['TotSignal']=TotSignal

#%% 

def pointpos(x):
    if x['TotSignal'] == 1:
        return x['High']+1e-3
    elif x['TotSignal'] == 2:
        return x['Low']-1e-3
    else:
        return np.nan

df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

#%%

dfpl = df[1800:2200]

fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=1), name="EMA")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="Signal")
fig.show()
#%%