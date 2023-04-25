import pandas as pd

class Klines:
    def __init__(self, klines_list):
        self.klines_list = klines_list

    def klines_df(self) -> pd.DataFrame:
        timestamp = ["open_time", "close_time"]

        float_column = [
            "open",
            "high",
            "low",
            "close",
            "quote_asset_volume",
            "taker_buy_quote_asset_volume",
        ]

        int_column = ["volume", "number_of_trades", "taker_buy_base_asset_volume"]

        columns = (
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        )

        dataframe = pd.DataFrame(self.klines_list, columns=columns)

        dataframe[timestamp] = dataframe[timestamp].astype("datetime64[ms]")
        dataframe[float_column] = dataframe[float_column].astype(float)
        dataframe[int_column] = dataframe[int_column].astype(int)
        dataframe.set_index("open_time", inplace=True)
        return dataframe

    def get_df_to_csv(self, dataframe, name) -> None:
        str_name = f"{name}.csv"
        columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        dataframe.to_csv(
            f"model/data/{str_name}",
            index=True,
            header=columns,
            sep=";",
            decimal=".",
            encoding="utf-8",
        )

        return print(str_name + " has been saved")
