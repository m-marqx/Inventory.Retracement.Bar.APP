import pandas as pd

class Klines:
    def __init__(self, klines_list):
        self.klines_list = klines_list

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
