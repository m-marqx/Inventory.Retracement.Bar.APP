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
    df_filtered = pd.DataFrame(dataframe[['open','high','low','close']]) # Filter out the columns we don't need

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