#%%
from typing import Optional
import pandas as pd
import numpy as np
from ..indicators import moving_average

ma = moving_average.moving_average()

#%%
def process_data(profit, dataframe, length=20, lowestlow=1, tick_size=0.1):
    try:
        df_filtered = dataframe[["open", "high", "low", "close"]].copy()
    except KeyError:
        df_filtered = dataframe[["Open", "High", "Low", "Close"]].copy()
        df_filtered.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"},
            inplace=True,
        )

    df_filtered["open"] = df_filtered["open"].astype(float)
    df_filtered["high"] = df_filtered["high"].astype(float)
    df_filtered["low"] = df_filtered["low"].astype(float)
    df_filtered["close"] = df_filtered["close"].astype(float)
    open_price = df_filtered["open"]
    high = df_filtered["high"]
    low_price = df_filtered["low"]
    close_price = df_filtered["close"]

    ema = ma.ema(close_price, length)
    df_filtered["ema"] = ema
    df_filtered["uptrend"] = np.where(close_price >= df_filtered["ema"], True, False)

    is_bullish = df_filtered["uptrend"] == True

    candle_amplitude = high - low_price
    candle_downtail = np.minimum(open_price, close_price) - low_price  # type: ignore
    candle_uppertail = high - np.maximum(open_price, close_price)

    # Analyze the downtail and uptail of the candle and assign a value to the IRB_Condition column based on the decimal value of the downtail or uptail
    bullish_calculation = candle_uppertail / candle_amplitude
    bearish_calculation = candle_downtail / candle_amplitude

    df_filtered["IRB_Condition"] = np.where(
        is_bullish, bullish_calculation, bearish_calculation
    )
    irb_condition = df_filtered["IRB_Condition"] >= 0.45
    buy_condition = irb_condition & is_bullish

    df_filtered["Signal"] = np.where(buy_condition, 1, np.nan)
    df_filtered["Signal"].astype("float32")

    entry_price = df_filtered["high"].shift(1) + tick_size
    target = df_filtered["high"].shift(1) + (candle_amplitude.shift(1) * profit)

    # Stop Loss is the lowest low of the last X candles
    stop_loss = df_filtered["low"].rolling(window=lowestlow).min().shift(1) - tick_size

    # If the lowest low is NaN, fill it with the cumulative minimum
    stop_loss = stop_loss.fillna(df_filtered["low"].cummin())

    df_filtered["Entry_Price"] = np.where(buy_condition, entry_price, np.nan)
    df_filtered["Take_Profit"] = np.where(buy_condition, target, np.nan)
    df_filtered["Stop_Loss"] = np.where(buy_condition, stop_loss, np.nan)

    return df_filtered
# %%

def IRB_strategy(df):
    dataframe = df.copy()
    dataframe.reset_index(inplace=True)
    dataframe["Close Position"] = False

    for index in range(1, dataframe.shape[0]):
        prev_index = index - 1
        if (dataframe["Signal"].iloc[prev_index] == 1) & (
            dataframe["Close Position"].iloc[index] == False
        ):
            dataframe.loc[index, "Signal"] = dataframe["Signal"].iloc[prev_index]
            dataframe.loc[index, "Entry_Price"] = dataframe["Entry_Price"].iloc[prev_index]
            dataframe.loc[index, "Take_Profit"] = dataframe["Take_Profit"].iloc[prev_index]
            dataframe.loc[index, "Stop_Loss"] = dataframe["Stop_Loss"].iloc[prev_index]

            if (dataframe["high"].iloc[index] > dataframe["Take_Profit"].iloc[index]) ^ (dataframe["low"].iloc[index] < dataframe["Stop_Loss"].iloc[index]):
                dataframe.loc[index, "Close Position"] = True
                dataframe.loc[index, "Signal"] = -1

    return dataframe


# %%
def calculate_results(dataframe, check_error=False):
    is_close_position = dataframe["Close Position"] == True
    is_take_profit = dataframe["high"] > dataframe["Take_Profit"]
    is_stop_loss = dataframe["low"] < dataframe["Stop_Loss"]

    profit = dataframe["Take_Profit"] - dataframe["Entry_Price"]
    loss = dataframe["Stop_Loss"] - dataframe["Entry_Price"]

    dataframe["Result"] = 0
    dataframe["Result"] = np.where(
        is_close_position & is_take_profit, profit, dataframe["Result"]
    )
    dataframe["Result"] = np.where(
        is_close_position & is_stop_loss, loss, dataframe["Result"]
    )
    dataframe["Cumulative_Result"] = dataframe["Result"].cumsum()

    if check_error:
        dataframe["Signal_Shifted"] = dataframe["Signal"].shift(1)
        dataframe["Check_Error"] = np.where(
            (pd.isnull(dataframe["Signal"])) & (dataframe["Signal_Shifted"] == 1),
            True,
            False,
        )
        dataframe["Check_Error"] = np.where(
            (
                pd.isnull(dataframe["Signal_Shifted"]) & dataframe["Close Position"]
                == True
            ),
            True,
            dataframe["Check_Error"],
        )
    if dataframe[dataframe["Check_Error"] == True].shape[0] > 0:
        print("Error Found")


def calculate_fixed_pl_results(dataframe, profit, loss, check_error=False):
    is_close_position = dataframe["Close Position"] == True
    is_take_profit = dataframe["high"] > dataframe["Take_Profit"]
    is_stop_loss = dataframe["low"] < dataframe["Stop_Loss"]

    dataframe["Result"] = 0
    dataframe["Result"] = np.where(
        is_close_position & is_take_profit, profit, dataframe["Result"]
    )
    dataframe["Result"] = np.where(
        is_close_position & is_stop_loss, -loss, dataframe["Result"]
    )
    dataframe["Cumulative_Result"] = dataframe["Result"].cumsum()

    if check_error:
        dataframe["Signal_Shifted"] = dataframe["Signal"].shift(1)
        dataframe["Check_Error"] = np.where(
            (pd.isnull(dataframe["Signal"])) & (dataframe["Signal_Shifted"] == 1),
            True,
            False,
        )
        dataframe["Check_Error"] = np.where(
            (
                pd.isnull(dataframe["Signal_Shifted"]) & dataframe["Close Position"]
                == True
            ),
            True,
            dataframe["Check_Error"],
        )
    if dataframe[dataframe["Check_Error"] == True].shape[0] > 0:
        print("Error Found")


def run_IRB_model(
    profit, length=20, dataframe=Optional[pd.DataFrame], csv_file=Optional[str]
):
    if csv_file is not Optional[str]:
        df = pd.read_csv(
            f"{csv_file}", sep=";", decimal=".", encoding="utf-8", index_col="open_time"
        )
    elif dataframe is not Optional[pd.DataFrame]:
        df = dataframe.copy()
    else:
        raise ValueError("Either 'dataframe' or 'csv_name' must be provided")

    df_filtered = process_data(profit, df, length)
    df_backtest = IRB_strategy(df_filtered)

    calculate_results(df_backtest, check_error=True)
    return df_backtest

#%%
def run_IRB_model_fixed(
    target,
    profit,
    loss,
    length=20,
    dataframe=Optional[pd.DataFrame],
    csv_file=Optional[str],
):
    if csv_file is not Optional[str]:
        df = pd.read_csv(
            f"{csv_file}", sep=";", decimal=".", encoding="utf-8", index_col="open_time"
        )
    elif dataframe is not Optional[pd.DataFrame]:
        df = dataframe.copy()
    else:
        raise ValueError("Either 'dataframe' or 'csv_file' must be provided")

    df_filtered = process_data(target, df, length)
    df_backtest = IRB_strategy(df_filtered)

    calculate_fixed_pl_results(df_backtest, profit, loss, check_error=True)
    return df_backtest

#%%
def EM_Calculation(dataframe):
    df = dataframe["Result"].copy()

    df["Gain Count"] = np.where(df["Result"] > 0, 1, 0)
    df["Loss Count"] = np.where(df["Result"] < 0, 1, 0)

    df["Gain Count"] = df["Gain Count"].cumsum()
    df["Loss Count"] = df["Loss Count"].cumsum()

    df["Mean Gain"] = df.query("Result > 0")["Result"].expanding().mean()
    df["Mean Loss"] = df.query("Result < 0")["Result"].expanding().mean()

    df["Mean Gain"].fillna(method="ffill", inplace=True)
    df["Mean Loss"].fillna(method="ffill", inplace=True)

    df["Total Gain"] = np.where(df["Result"] > 0, df["Result"], 0).cumsum()
    df["Total Loss"] = np.where(df["Result"] < 0, df["Result"], 0).cumsum()

    df["Total Trade"] = df["Gain Count"] + df["Loss Count"]
    df["Win Rate"] = df["Gain Count"] / df["Total Trade"]
    df["Loss Rate"] = df["Loss Count"] / df["Total Trade"]

    # EM
    df["EM_Gain"] = df["Mean Gain"] * df["Win Rate"]
    df["EM_Loss"] = df["Mean Loss"] * df["Loss Rate"]
    df["EM"] = df["EM_Gain"] - abs(df["EM_Loss"])
    df = df.query("Result != 0")

    return df
