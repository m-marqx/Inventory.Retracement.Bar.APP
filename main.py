from model.strategy.IRB import IRB_Strategy
from controller.binance_api import futures_API

fAPI = futures_API()
BTC = fAPI.get_all_futures_klines_df("BTCUSD_PERP", "2h", 7_200_000)

strategy = IRB_Strategy(BTC)
strategy = strategy.run_IRB_model(2,20,0.1)