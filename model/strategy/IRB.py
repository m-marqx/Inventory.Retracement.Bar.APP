
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

def get_all_futures_klines_df(symbol, interval, intervalms):
    klines_list = fapi.get_All_Klines(interval, intervalms, symbol=symbol)
    dataframe = pd.DataFrame(klines_list,columns=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
    dataframe.set_index('open_time', inplace=True)
    return dataframe

def klines_df_to_csv(dataframe, symbol, interval):
    str_name = str(symbol) + '-' + str(interval) + '.csv'
    dataframe.to_csv(str_name, index=True, header=['open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'], sep=';', decimal='.', encoding='utf-8')
    return print(str_name+' has been saved')

#%% 
def process_data(profit, dataframe, length=20):
    try:
        df_filtered = pd.DataFrame(dataframe[['open','high','low','close']]) # Filter out the columns we don't need
    except:
        df_filtered = pd.DataFrame(dataframe[['Open','High','Low','Close']]) # Filter out the columns we don't need
        df_filtered = df_filtered.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})

    Open = df_filtered['open']
    High = df_filtered['high']
    Low = df_filtered['low']
    Close = df_filtered['close']
    ema = ma.ema(Close, length)
    df_filtered['ema'] = ema
    df_filtered['market_trend'] = np.where(Close >= df_filtered['ema'], "Bullish", "Bearish")

    is_bullish = df_filtered['market_trend'] == "Bullish"

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
    df_filtered['Entry_Price'] = np.where(buy_condition, df_filtered['high'].shift(1), np.nan)
    df_filtered['Take_Profit'] = np.where(buy_condition, (candle_amplitude.shift(1) * profit) + df_filtered['high'].shift(1), np.nan)
    df_filtered['Stop_Loss'] = np.where(buy_condition, df_filtered['low'].shift(1) - 1, np.nan)

    return df_filtered
#%%

profit = 2
df = pd.read_csv('BTCUSD_PERP-2h.csv', sep=';', decimal='.', encoding='utf-8', index_col='open_time')
df_filtered = process_data(profit, df,20)

#%%
import time

def IRB_strategy(df):
    dataframe = df.copy()
    dataframe.reset_index(inplace=True)
    dataframe['Close Position'] = False
    
    print('Loop Started')
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

df1 = IRB_strategy(df_filtered)

#%%
def calculate_results(dataframe, check_method=False):
    is_close_position = dataframe['Close Position'] == True
    is_take_profit = dataframe['high'] > dataframe['Take_Profit']
    is_stop_loss = dataframe['low'] < dataframe['Stop_Loss']
    
    profit = dataframe['Take_Profit'] - dataframe['Entry_Price']
    loss = dataframe['Stop_Loss'] - dataframe['Entry_Price']
    
    dataframe['Result'] = 0 
    dataframe['Result'] = np.where(is_close_position & is_take_profit, profit, dataframe['Result'])
    dataframe['Result'] = np.where(is_close_position & is_stop_loss, loss, dataframe['Result'])
    dataframe['Cumulative_Result'] = dataframe['Result'].cumsum()

    if check_method:
        dataframe['Signal_Shifted'] = dataframe['Signal'].shift(1) == 1
        dataframe['Check_Method'] = np.where((pd.isnull(dataframe['Signal'])) & (dataframe['Signal_Shifted'] == 1), "ERROR", 0)
        dataframe['Check_Method'] = np.where((pd.isnull(dataframe['Signal_Shifted']) & dataframe['Close Position'] == True), "ERROR", dataframe['Check_Method'])

calculate_results(df1, check_method=True)

df1['Cumulative_Result'].plot()