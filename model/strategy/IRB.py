import pandas as pd
import sys
import os

# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the desired directory
controller_dir = os.path.join(current_dir, '..', '..', 'controller')

# Add the desired directory to the Python path
sys.path.insert(0,controller_dir)

import binance_api as bAPI
# Import the class

# Create an instance of the class
API_KEY = os.environ['API_KEY']
SECRET_KEY = os.environ['SECRET_KEY']

fapi = bAPI.futures_API(API_KEY, SECRET_KEY)

def get_all_futures_klines_df(symbol, interval, intervalms):
    klines_list = fapi.get_All_Klines(interval, intervalms, symbol=symbol)
    dataframe = pd.DataFrame(klines_list,columns=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
    dataframe.set_index('open_time', inplace=True)
    return dataframe

def klines_df_to_csv(dataframe, symbol, interval):
    str_name = str(symbol) + '-' + str(interval) + '.csv'
    dataframe.to_csv(str_name, index=True, header=['open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'], sep=';', decimal='.', encoding='utf-8')
    return print(str_name+'has been saved')