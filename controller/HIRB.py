# Importing necessary libraries
import yfinance as yf
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go

# Downloading historical currency data of Euro-US dollar exchange rate
df = yf.download(tickers='EURUSD=X', period='59d', interval='15m')

# Resetting the index and droping the previous one
df.reset_index(drop=True, inplace=True)

# Calculating Exponential Moving Average (EMA) with a length of 20
df["EMA"] = ta.ema(df.Close, length=20)

# Calculating the slope of EMA with a window size of 20
backrollingN = 20
df['slopeEMA'] = df['EMA'].diff(periods=1)
df['slopeEMA'] = df['slopeEMA'].rolling(window=backrollingN).mean()

# Creating an empty list to store the signals
TotSignal = [0] * len(df)

# Setting limits for slope and percent
slopelimit = 5e-5
percentlimit = 0.45

# Initializing previous signal as None
previous_signal = None

# Looping through the dataframe to generate the signals
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

# Adding the signals to the dataframe
df['TotSignal']=TotSignal

# Function to calculate the position of the signal points
def pointpos(x):
    if x['TotSignal'] == 1:
        return x['High']+1e-3
    elif x['TotSignal'] == 2:
        return x['Low']-1e-3
    else:
        return np.nan

# Applying the function to the dataframe
df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

# Selecting a portion of the dataframe for visualization
dfpl = df[1800:2200]

# Creating a Plotly figure with candlestick chart and EMA line
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
