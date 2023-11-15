## PT-BR
Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/Hoffman-IRB/blob/master/README%20-%20BR.md)

## Initial Idea

The repository aims to create a model that collects data from Binance, the largest international exchange, using the `python-binance` library. Afterward, the data is processed to obtain the results of the `Inventory Retracement Bar` strategy, developed by `Rob Hoffman`. The strategy showed promise, as the author claims it has brought victories in various trading competitions. However, it was observed that the strategy's default settings yielded unsatisfactory results in the cryptocurrency market.

## Objective

Create a model where the signal occurs when the wick of the candle is equal to or greater than 45% of the body of the candle.

![Signal](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/3e1f7810-19f6-45b3-ba9f-b7a940463c77)

## Entry Signal

By default, a 20-period Simple Moving Average is used to indicate the trend. Although the author suggests seeking trades where the moving average has a 45-degree slope, this idea was discarded due to the subjectivity of the angle when zooming in on the chart.

![62-degree](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/595ce7f2-3440-49b8-933e-78ad29c953f7)

![45-degree](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/946d999d-75a1-4038-8561-5c05145dc7f6)

## Trading

The entry occurs when the candle breaks the maximum value of the candle that generated the signal. Although this reduces the total number of entries, it's not an issue due to the high number of signals in the setup.

![Start](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/9b93b458-6a31-4a55-a7e0-f12945c5cd33)

The target value depends on the chosen `payoff`, which is the amplitude of the signal candle, and the stop value should be at the signal candle's minimum or the lowest value of the last N candles.
This example uses a `Payoff` of 2 and the stop at the minimum of the signal candle.

![End](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/8684d8f7-323e-4089-94b1-57b3420d8e03)

## How to Use

To enhance result visualization, a Dashboard has been created using `Dash`, from the same team behind `Plotly`. The Dashboard comprises 2 pages: `homepage` and `backtest`, each with a specific objective. Both pages aim to provide users with a simple and easy way to customize and visualize results with different parameter values for the strategy.

Visit the [homepage](https://github.com/m-marqx/Hoffman-IRB/tree/master/view/dashboard/pages/home) or [backtest](https://github.com/m-marqx/Hoffman-IRB/blob/master/view/dashboard/pages/backtest/) for detailed information on using each page of the Dashboard.
