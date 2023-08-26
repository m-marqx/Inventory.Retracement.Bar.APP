## Objective 🎯

The main objective of this page is to provide a user-friendly interface for customizing and visualizing the results of a backtest that uses GridSearch (also known as brute force) for identifying two key points:

1. The parameters that make the strategy profitable.
2. The **best parameter values** for achieving profit in this strategy.

These two points help identify the ideal range of values for achieving theoretical success with the IRB strategy.

Similar to the homepage, this page also aims to offer a high degree of customization and therefore includes almost all menus from the homepage: `Get Data`, `Modify Indicator Parameters`, `Modify Strategy Parameters`, `Modify Result Parameters`, along with the addition of the `Modify Hardware Parameters` menu.

## Data 📊

The `Get Data` menu's objective is to allow the user to choose the type of market from which they want to retrieve data:

- `Spot, Futures, Mark Price`: The user should select the Timeframe and Symbol for the data they wish to obtain.

![Data](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/a8cf8bb6-3c92-451d-a202-36305206446c)


- `Custom`: If the user already has the data, they can select the `Custom` option, which supports files in `parquet` and `csv` formats.

![Custom](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/5dc6236b-1c54-46d6-a3f9-851e62645121)


## Indicators 💹

In the `Modify Indicator Parameters` menu, only the EMA indicator is available for testing. A Grid Search will be conducted with all possible values of the `Minimum EMA Length` to the `Maximum EMA Length`.

![Indicators](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/c75fb6f5-1a39-494b-9589-97f9e4a673bc)


## Strategy ♟️

The `Modify Strategy Parameters` menu serves to define the minimum and maximum values for `Lowest Low`, `Payoff`, and `Wick Percentage`, and it includes all possible combinations in the Grid Search.

- `Lowest Low`: Specifies where the stop loss will be placed, defaulting to the current candle's low.
- `Tick Size`: Determines the value added to the entry price or subtracted from the stop loss.
- `Payoff`: Sets the target location, defaulting to `2` times the signal candle.
- `Wick Percentage`: Sets the minimum acceptable percentage for the signal candle's wick.


![Strategy](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/8b6aac60-7dcb-4750-932c-6c2a1f1c976d)


## Result 🪙

The `Modify Result Parameters` menu aims to modify risk management parameters and allows the user to select multiple options. The submenu `Modify Result Settings` is used to configure how results are displayed and whether the user wants to simulate returns in USD or the trading currency. Considering that the strategy might create a graph that's difficult to visualize due to a high number of displayed lines, the user can decide whether to show only positive results or all results, including those that resulted in losses.

![Result](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/db27da06-c1c9-4aa9-8a7b-74b3a39049b1)


## Hardware

The `Modify Hardware Parameters` menu's primary objective is to allow the user to modify the number of CPU cores or use the GPU for backtest processing. To prevent the user from selecting the GPU option without the necessary configurations or without having one, the PyTorch method `.cuda.device_count()` is used. If the user has a GPU, they can modify the options `GPU Quantity` and `GPU Workers`, and they will be able to select the `GPU` option.

GPU not found/configured:

![GPU Not Found](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/e50ae21e-b593-4432-8825-9337b3eb2824)

GPU found and configured:

![GPU Found](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/a106e748-8e5c-4535-ad15-c810c305679b)
