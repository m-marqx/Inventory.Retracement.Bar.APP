from controller.future_API import FuturesAPI
from model.strategy.params.indicators_params import (
    EmaParams,
    MACDParams,
    CCIParams,
)
from model.strategy.params.strategy_params import (
    IrbParams,
    IndicatorsParams,
)

from model.strategy.strategy import BuilderStrategy
from model.strategy.indicators import BuilderSource
# %%

FAPI = FuturesAPI()
BTC = FAPI.get_all_futures_klines_df("BTCUSD_PERP", "2h")

# %%
ema_params = EmaParams()
macd_params = MACDParams()
cci_params = CCIParams()


irb_params = IrbParams(wick_percentage=0.02)
indicators_params = IndicatorsParams()

#%%
df = (
    BuilderSource(
        BTC,
    )
    .set_EMA_params(ema_params)
    .set_ema()
    .set_MACD_params(macd_params)
    .set_macd()
    .set_CCI_params(cci_params)
    .set_cci()
    .execute()
)
#%%
df = (
    BuilderStrategy(
        df,
    )
    .set_trend_params(indicators_params)
    .set_trend()
    .set_irb_params(irb_params)
    .get_irb_signals()
    .calculate_irb_signals()
    .calculateResults()
    .execute()
)
# %%
df
# %%
