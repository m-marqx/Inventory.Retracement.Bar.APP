#%%
from controller.future_API import futuresAPI
from model.strategy.params.indicators_params import (
    EMA_params,
    MACD_params,
    CCI_params,
)
from model.strategy.params.strategy_params import (
    irb_params,
    indicators_params,
)

from model.strategy.strategy import builder_strategy
from model.strategy.indicators import builder_source
# %%

fAPI = futuresAPI()
BTC = fAPI.get_all_futures_klines_df("BTCUSD_PERP", "2h")

# %%
ema_params = EMA_params()
MACD_params = MACD_params()
CCI_params = CCI_params()


irb_params = irb_params()
indicators_params = indicators_params()

#%%
df = (
    builder_source(
        BTC,
    )
    .set_EMA_params(ema_params)
    .set_ema()
    .set_MACD_params(MACD_params)
    .set_macd()
    .set_CCI_params(CCI_params)
    .set_cci()
    .execute()
)
#%%
df = (
    builder_strategy(
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
