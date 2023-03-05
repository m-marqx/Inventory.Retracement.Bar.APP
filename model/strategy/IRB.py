
#%% 
import pandas as pd
import numpy as np
import sys
import os

# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the desired directory
controller_dir = os.path.join(current_dir, '..', '..', 'controller')
indicators_dir = os.path.join(current_dir, '..', 'indicators')

# Add the desired directory to the Python path
sys.path.insert(0,controller_dir)
sys.path.insert(0,indicators_dir)
# %% 
import binance_api as bAPI
import moving_average
# Import the class

# Create an instance of the class
API_KEY = os.environ['API_KEY']
SECRET_KEY = os.environ['SECRET_KEY']

fapi = bAPI.futures_API(API_KEY, SECRET_KEY)
ma = moving_average.moving_average()

#%% 
def process_data(profit, dataframe, length=20):
    try:
        df_filtered = pd.DataFrame(dataframe[['open','high','low','close']]) # Filter out the columns we don't need
    except:
        df_filtered = pd.DataFrame(dataframe[['Open','High','Low','Close']]) # Filter out the columns we don't need
        df_filtered = df_filtered.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})

    df_filtered['open'] = df_filtered['open'].astype(float)
    df_filtered['high'] = df_filtered['high'].astype(float)
    df_filtered['low'] = df_filtered['low'].astype(float)
    df_filtered['close'] = df_filtered['close'].astype(float)
    Open = df_filtered['open']
    High = df_filtered['high']
    Low = df_filtered['low']
    Close = df_filtered['close']
    ema = ma.ema(Close, length)
    df_filtered['ema'] = ema
    df_filtered['uptrend'] = np.where(Close >= df_filtered['ema'], True, False)

    is_bullish = df_filtered['uptrend'] == True

    candle_amplitude = High - Low
    candle_downtail = np.minimum(Open, Close) - Low # type: ignore
    candle_uppertail = High - np.maximum(Open, Close)

# Analyze the downtail and uptail of the candle and assign a value to the IRB_Condition column based on the decimal value of the downtail or uptail
    bullish_calculation = candle_uppertail / candle_amplitude
    bearish_calculation = candle_downtail / candle_amplitude
    
    df_filtered['IRB_Condition'] = np.where(is_bullish, bullish_calculation, bearish_calculation)
    irb_condition = df_filtered['IRB_Condition'] >= 0.45
    buy_condition = irb_condition & is_bullish
    
    df_filtered['Signal'] = np.where(buy_condition, 1, np.nan)
    df_filtered['Signal'].astype('float32')
    df_filtered['Entry_Price'] = np.where(buy_condition, df_filtered['high'].shift(1), np.nan)
    df_filtered['Take_Profit'] = np.where(buy_condition, (candle_amplitude.shift(1) * profit) + df_filtered['high'].shift(1), np.nan)
    df_filtered['Stop_Loss'] = np.where(buy_condition, df_filtered['low'].shift(1) - 1, np.nan)

    return df_filtered
#%%

import time

def IRB_strategy(df):
    dataframe = df.copy()
    dataframe.reset_index(inplace=True)
    dataframe['Close Position'] = False
    
    for x in range(1, dataframe.shape[0]):
        if (dataframe['Signal'].iloc[x-1] == 1) & (dataframe['Close Position'].iloc[x] == False):
            dataframe.loc[x,'Signal'] = dataframe['Signal'].iloc[x-1]
            dataframe.loc[x,'Entry_Price'] = dataframe['Entry_Price'].iloc[x-1]	
            dataframe.loc[x,'Take_Profit'] = dataframe['Take_Profit'].iloc[x-1]
            dataframe.loc[x,'Stop_Loss'] = dataframe['Stop_Loss'].iloc[x-1]

            if (dataframe['high'].iloc[x] > dataframe['Take_Profit'].iloc[x]) ^ (dataframe['low'].iloc[x] < dataframe['Stop_Loss'].iloc[x]):
                dataframe.loc[x,'Close Position'] = True
                dataframe.loc[x,'Signal'] = -1

    return dataframe

#%%
def calculate_results(dataframe, check_error=False):
    is_close_position = dataframe['Close Position'] == True
    is_take_profit = dataframe['high'] > dataframe['Take_Profit']
    is_stop_loss = dataframe['low'] < dataframe['Stop_Loss']
    
    profit = dataframe['Take_Profit'] - dataframe['Entry_Price']
    loss = dataframe['Stop_Loss'] - dataframe['Entry_Price']
    
    dataframe['Result'] = 0 
    dataframe['Result'] = np.where(is_close_position & is_take_profit, profit, dataframe['Result'])
    dataframe['Result'] = np.where(is_close_position & is_stop_loss, loss, dataframe['Result'])
    dataframe['Cumulative_Result'] = dataframe['Result'].cumsum()

    if check_error:
        dataframe['Signal_Shifted'] = dataframe['Signal'].shift(1)
        dataframe['Check_Error'] = np.where((pd.isnull(dataframe['Signal'])) & (dataframe['Signal_Shifted'] == 1), True, False)
        dataframe['Check_Error'] = np.where((pd.isnull(dataframe['Signal_Shifted']) & dataframe['Close Position'] == True), True, dataframe['Check_Error'])
    if df_backtest[df_backtest['Check_Error'] == True].shape[0] > 0:
        print('Error Found')

def calculate_fixed_pl_results(dataframe, profit, loss, check_error=False):
    is_close_position = dataframe['Close Position'] == True
    is_take_profit = dataframe['high'] > dataframe['Take_Profit']
    is_stop_loss = dataframe['low'] < dataframe['Stop_Loss']
    
    dataframe['Result'] = 0 
    dataframe['Result'] = np.where(is_close_position & is_take_profit, profit, dataframe['Result'])
    dataframe['Result'] = np.where(is_close_position & is_stop_loss, -loss, dataframe['Result'])
    dataframe['Cumulative_Result'] = dataframe['Result'].cumsum()

    if check_error:
        dataframe['Signal_Shifted'] = dataframe['Signal'].shift(1)
        dataframe['Check_Error'] = np.where((pd.isnull(dataframe['Signal'])) & (dataframe['Signal_Shifted'] == 1), True, False)
        dataframe['Check_Error'] = np.where((pd.isnull(dataframe['Signal_Shifted']) & dataframe['Close Position'] == True), True, dataframe['Check_Error'])
    if df_backtest[df_backtest['Check_Error'] == True].shape[0] > 0:
        print('Error Found')

profit = 2
try:
    df = pd.read_csv('BTCUSD_PERP-2h.csv', sep=';', decimal='.', encoding='utf-8', index_col='open_time')
except:
    df = fapi.get_all_futures_klines_df('BTCUSD_PERP', '2h', 7200000)
    fapi.klines_df_to_csv(df, 'BTCUSD_PERP', '2h')

df_filtered = process_data(profit, df,20)
df_backtest = IRB_strategy(df_filtered)

calculate_results(df_backtest, check_error=True)
df_backtest['Cumulative_Result'].plot()

df_backtest2 = IRB_strategy(df_filtered)
calculate_fixed_pl_results(df_backtest2, 200, 100, check_error=True)
df_backtest2['Cumulative_Result'].plot()