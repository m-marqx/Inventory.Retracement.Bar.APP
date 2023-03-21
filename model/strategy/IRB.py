#%%
from typing import Optional
import pandas as pd
import numpy as np
from ..indicators import moving_average

ma = moving_average.moving_average()

#%%
def process_data(profit, dataframe, length=20, lowestlow=1, tick_size=0.1):
    columns = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    try:
        df_filtered = dataframe[columns.values()].copy()
    except KeyError:
        df_filtered = dataframe[columns.keys()].copy()
        df_filtered.rename(
            columns=columns,
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
    df_filtered["uptrend"] = close_price >= df_filtered["ema"]

    candle_amplitude = high - low_price
    candle_downtail = np.minimum(open_price, close_price) - low_price
    candle_uppertail = high - np.maximum(open_price, close_price)

    # Analyze the downtail and uptail of the candle
    # Assign a value to the IRB_Condition column based on the value of the wick
    bullish_calculation = candle_uppertail / candle_amplitude
    bearish_calculation = candle_downtail / candle_amplitude

    df_filtered["IRB_Condition"] = np.where(
        df_filtered["uptrend"], bullish_calculation, bearish_calculation
    )
    irb_condition = df_filtered["IRB_Condition"] >= 0.45
    buy_condition = irb_condition & df_filtered["uptrend"]

    df_filtered["Signal"] = np.where(buy_condition, 1, np.nan)
    df_filtered["Signal"].astype("float32")

    entry_price = df_filtered["high"].shift(1) + tick_size
    target = df_filtered["high"].shift(1) + (candle_amplitude.shift(1) * profit)

    # Stop Loss is the lowest low of the last X candles
    stop_loss = df_filtered["low"].rolling(lowestlow).min().shift() - tick_size

    # If the lowest low is NaN, fill it with the cumulative minimum
    stop_loss = stop_loss.fillna(df_filtered["low"].cummin())

    df_filtered["Entry_Price"] = np.where(buy_condition, entry_price, np.nan)
    df_filtered["Take_Profit"] = np.where(buy_condition, target, np.nan)
    df_filtered["Stop_Loss"] = np.where(buy_condition, stop_loss, np.nan)

    return df_filtered


# TODO2: aqui acaba a primeira parte da classe.
# %%


def IRB_strategy(dataframe):
    signal = dataframe["Signal"].values
    entry_price = dataframe["Entry_Price"].values
    take_profit = dataframe["Take_Profit"].values
    stop_loss = dataframe["Stop_Loss"].values
    high = dataframe["high"].values
    low = dataframe["low"].values
    close_position = np.zeros(len(dataframe), dtype=bool)

    for index in range(1, len(dataframe)):
        prev_index = index - 1
        signal_condition = signal[prev_index] == 1
        open_position = ~close_position[index]
        if signal_condition & open_position:
            signal[index] = signal[prev_index]
            entry_price[index] = entry_price[prev_index]
            take_profit[index] = take_profit[prev_index]
            stop_loss[index] = stop_loss[prev_index]
            profit = high[index] > take_profit[index]
            loss = low[index] < stop_loss[index]

            if profit ^ loss:
                close_position[index] = True
                signal[index] = -1

    data_frame = pd.DataFrame(
        {
            "Signal": signal,
            "Entry_Price": entry_price,
            "Take_Profit": take_profit,
            "Stop_Loss": stop_loss,
            "Close Position": close_position,
            "high": high,
            "low": low,
        }
    )

    return data_frame


# %%
def check_error(dataframe):
    columns = ["Signal", "Close Position"]
    data_frame = dataframe[columns].copy()

    data_frame["Signal_Shifted"] = data_frame["Signal"].shift(1)

    is_null_signal = data_frame["Signal"].isnull()
    is_null_signal_shift = data_frame["Signal_Shifted"].isnull()
    has_signal_error = is_null_signal & (data_frame["Signal_Shifted"] == 1)
    has_close_error = is_null_signal_shift & data_frame["Close Position"]

    has_error = has_signal_error | has_close_error

    if has_error.any():
        print("Error Found")

    data_frame["Error"] = has_error
    return data_frame


# %%


def calculate_results(dataframe, verify_error=True):
    columns = [
        "Signal",
        "Entry_Price",
        "Take_Profit",
        "Stop_Loss",
        "high",
        "low",
        "Close Position",
    ]
    data_frame = dataframe[columns].copy()
    is_close_position = data_frame["Close Position"]
    is_take_profit = data_frame["high"] > data_frame["Take_Profit"]
    is_stop_loss = data_frame["low"] < data_frame["Stop_Loss"]

    profit = data_frame["Take_Profit"] - data_frame["Entry_Price"]
    loss = data_frame["Stop_Loss"] - data_frame["Entry_Price"]

    data_frame["Result"] = 0
    data_frame["Result"] = np.where(
        is_close_position & is_take_profit, profit, data_frame["Result"]
    )
    data_frame["Result"] = np.where(
        is_close_position & is_stop_loss, loss, data_frame["Result"]
    )
    data_frame["Cumulative_Result"] = data_frame["Result"].cumsum()

    if verify_error:
        check_error(data_frame)

    return data_frame


# %%
def calculate_fixed_pl_results(dataframe, profit, loss, verify_error=False):
    columns = [
        "Signal",
        "Entry_Price",
        "Take_Profit",
        "Stop_Loss",
        "high",
        "low",
        "Close Position",
    ]
    data_frame = dataframe[columns].copy()
    is_close_position = data_frame["Close Position"]
    is_take_profit = data_frame["high"] > data_frame["Take_Profit"]
    is_stop_loss = data_frame["low"] < data_frame["Stop_Loss"]

    data_frame["Result"] = 0
    data_frame["Result"] = np.where(
        is_close_position & is_take_profit, profit, data_frame["Result"]
    )
    data_frame["Result"] = np.where(
        is_close_position & is_stop_loss, -loss, data_frame["Result"]
    )
    data_frame["Cumulative_Result"] = data_frame["Result"].cumsum()

    if verify_error:
        check_error(data_frame)

    return data_frame


# %%


def run_IRB_model(
    profit, length=20, dataframe=Optional[pd.DataFrame], csv_file=Optional[str]
):
    if csv_file is not Optional[str]:
        data_frame = pd.read_csv(
            f"{csv_file}", sep=";", decimal=".", encoding="utf-8", index_col="open_time"
        )
    elif dataframe is not Optional[pd.DataFrame]:
        data_frame = dataframe.copy()
    else:
        raise ValueError("Either 'dataframe' or 'csv_name' must be provided")

    df_filtered = process_data(profit, data_frame, length)
    df_strategy = IRB_strategy(df_filtered)
    df_backtest = calculate_results(df_strategy, True)

    return df_backtest


# %%
def run_IRB_model_fixed(
    target,
    profit,
    loss,
    length=20,
    dataframe=Optional[pd.DataFrame],
    csv_file=Optional[str],
):
    if csv_file is not Optional[str]:
        data_frame = pd.read_csv(
            f"{csv_file}", sep=";", decimal=".", encoding="utf-8", index_col="open_time"
        )
    elif dataframe is not Optional[pd.DataFrame]:
        data_frame = dataframe.copy()
    else:
        raise ValueError("Either 'dataframe' or 'csv_file' must be provided")

    df_filtered = process_data(target, data_frame, length)
    df_strategy = IRB_strategy(df_filtered)
    df_backtest = calculate_fixed_pl_results(df_strategy, profit, loss, True)

    return df_backtest


# %%
# TODO6: Esse aqui eu acho que poderia fazer em um arquivo separado fora de strategy inclusive, mas ele se encaixa no 50 tons de result.
def EM_Calculation(dataframe):
    data_frame = dataframe["Result"].copy()

    data_frame["Gain Count"] = np.where(data_frame["Result"] > 0, 1, 0)
    data_frame["Loss Count"] = np.where(data_frame["Result"] < 0, 1, 0)

    data_frame["Gain Count"] = data_frame["Gain Count"].cumsum()
    data_frame["Loss Count"] = data_frame["Loss Count"].cumsum()

    data_frame["Mean Gain"] = (
        data_frame.query("Result > 0")["Result"].expanding().mean()
    )
    data_frame["Mean Loss"] = (
        data_frame.query("Result < 0")["Result"].expanding().mean()
    )

    data_frame["Mean Gain"].fillna(method="ffill", inplace=True)
    data_frame["Mean Loss"].fillna(method="ffill", inplace=True)

    data_frame["Total Gain"] = np.where(
        data_frame["Result"] > 0, data_frame["Result"], 0
    ).cumsum()
    data_frame["Total Loss"] = np.where(
        data_frame["Result"] < 0, data_frame["Result"], 0
    ).cumsum()

    data_frame["Total Trade"] = data_frame["Gain Count"] + data_frame["Loss Count"]
    data_frame["Win Rate"] = data_frame["Gain Count"] / data_frame["Total Trade"]
    data_frame["Loss Rate"] = data_frame["Loss Count"] / data_frame["Total Trade"]

    # expected mathematical calculation
    em_gain = data_frame["Mean Gain"] * data_frame["Win Rate"]
    em_loss = data_frame["Mean Loss"] * data_frame["Loss Rate"]
    data_frame["EM"] = em_gain - abs(em_loss)
    data_frame = data_frame.query("Result != 0")

    return data_frame
