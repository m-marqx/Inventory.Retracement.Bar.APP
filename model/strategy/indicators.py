from model.indicators import CCI, MACD, MovingAverage
from model.utils import CleanData, BaseStrategy
from model.strategy.params import (
    EmaParams,
    MACDParams,
    CCIParams
)


class CalculateEma(BaseStrategy):
    def __init__(self, dataframe, params: EmaParams):
        """
        Initialize the CalculateEma object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        params : EmaParams
            The parameters for EMA calculation.
        """
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.length = params.length

    def execute(self):
        """
        Execute the EMA calculation.

        Returns:
        --------
        pd.DataFrame
            The dataframe with EMA values.
        """

        ma = MovingAverage()
        self.ema = ma.ema(self.source, self.length)
        self.df_filtered["ema"] = self.ema
        return self.df_filtered


class CalculateMACD(BaseStrategy):
    def __init__(self, dataframe, params: MACDParams):
        """
        Initialize the CalculateMACD object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        params : MACDParams
            The parameters for MACD calculation.
        """
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.fast_length = params.fast_length
        self.slow_length = params.slow_length
        self.signal_length = params.signal_length

    def execute(self):
        """
        Execute the MACD calculation.

        Returns:
        --------
        pd.DataFrame
            The dataframe with MACD histogram values.
        """
        self.histogram = MACD(
            self.source,
            self.fast_length,
            self.slow_length,
            self.signal_length,
        ).get_histogram

        self.df_filtered["MACD_Histogram"] = self.histogram
        return self.df_filtered


class CalculateCCI(BaseStrategy):
    def __init__(self, dataframe, params: CCIParams):
        """
        Initialize the CalculateCCI object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        params : CCIParams
            The parameters for CCI calculation.
        """
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.params = params

    def execute(self):
        """
        Execute the CCI calculation.

        Returns:
        --------
        pd.DataFrame
            The dataframe with CCI values.
        """
        self.CCI = CCI(self.source, self.params.length)

        if self.params.ma_type == "sma":
            self.CCI.set_sma()
        if self.params.ma_type == "ema":
            self.CCI.set_ema()

        self.df_filtered["CCI"] = self.CCI.CCI(self.params.constant)["CCI"]
        return self.df_filtered


class BuilderSource(BaseStrategy):
    def __init__(self, dataframe):
        """
        Initialize the BuilderSource object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        """
        super().__init__(dataframe)
        self.df_filtered = CleanData(self.df_filtered).execute()

    def set_EMA_params(self, params: EmaParams = EmaParams()):
        """
        Set the parameters for EMA calculation.

        Parameters:
        -----------
        params : EmaParams, optional
            The parameters for EMA calculation.

        Returns:
        --------
        BuilderSource
            The BuilderSource object.
        """
        self.ema_params = params
        return self

    def set_ema(self):
        """
        Execute the EMA calculation and set the EMA values in the dataframe.

        Returns:
        --------
        BuilderSource
            The BuilderSource object.
        """
        CalculateEma(self.df_filtered, self.ema_params).execute()
        return self

    def set_CCI_params(self, params: CCIParams = CCIParams()):
        """
        Set the parameters for CCI calculation.

        Parameters:
        -----------
        params : CCIParams, optional
            The parameters for CCI calculation.

        Returns:
        --------
        BuilderSource
            The BuilderSource object.
        """
        self.cci_params = params
        return self

    def set_cci(self):
        """
        Execute the CCI calculation and set the CCI values in the dataframe.

        Returns:
        --------
        BuilderSource
            The BuilderSource object.
        """
        CalculateCCI(self.df_filtered, self.cci_params).execute()
        return self

    def set_MACD_params(self, params: MACDParams = MACDParams()):
        """
        Set the parameters for MACD calculation.

        Parameters:
        -----------
        params : MACDParams, optional
            The parameters for MACD calculation.

        Returns:
        --------
        BuilderSource
            The BuilderSource object.
        """
        self.macd_params = params
        return self

    def set_macd(self):
        """
        Execute the MACD calculation and set the MACD histogram values in the dataframe.

        Returns:
        --------
        BuilderSource
            The BuilderSource object.
        """
        CalculateMACD(self.df_filtered, self.macd_params).execute()
        return self

    def execute(self):
        """
        Execute the data cleaning and strategy calculations.

        Returns:
        --------
        pd.DataFrame
            The processed dataframe.
        """
        return self.df_filtered
