## PT-BR
Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/Hoffman-IRB/blob/master/view/dashboard/pages/ml/Readme%20-%20BR.md)

## Objective 🎯

This page has 3 objectives:

1. To be an interface where the user creates a Machine Learning model in a simplified way.
2. To allow a high degree of customization for the user with the menus: `Get Data`, `Modify Indicators`, `Modify Feature Parameters`, `Modify Model Parameters`, `Modify Data Split`, each menu with a specific responsibility.
3. To obtain model predictions.

## Data 📊

The `Get Data` menu aims to allow the user to choose which cryptocurrency the model will use for data, such as the `Symbol`, which should contain the name of the `pair` that the model will use in its database, and should be `exactly` as seen on the exchange.  

![Get Data](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/fc564c8b-138a-4420-8a5b-812ef31b6b94)

## Indicators 💹

The `Modify Indicators` menu allows selecting which indicator can be used, and if there is any additional configuration for that indicator, a menu for configuration will be added:

Without any additional configuration:

![indicators modify1](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/e8376dc4-a9b0-4358-9dde-d0f4b77b524d)

With some additional configuration:

![indicators modify2](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/33c02443-c988-4d9b-b2f5-15123a7338b9)

## Features ♟️

In terms of trading, features are metaphorically the model's `setup`. In other words, they are a set of rules for the model.

In this repository, the algorithm used is the `Gradient Boosting` from the `XGBoost` library.

All model features are numerical features. With this in mind, the 3 numerical inputs are the `Threshold Points` that the model will use to define the value range that will be used to transform each numerical feature into three binary features.

The `Feature Selection` menu allows the user to choose which binary feature will be used.

- The selected order of these features can impact the final result of the model.

![modify features](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/aea22d80-79a1-4938-a10f-5f6fa27b7eba)
## Model Parameters 🤖

Model parameters are values that define how the model will learn to classify binary features. Whenever there's a new configuration in the indicators or features, it's recommended to click the `Generate Parameters` button to search for the best values for the model.

Default values:

![default model params](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/1f1da426-47e4-4f5e-b5d0-bf2145fdcb20)

Generated values:

![new model params](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/b535c200-589c-4cb9-8502-c15d9a6b2e7a)

## Data Split 📈📉📈

When it comes to Machine Learning, the dataset is typically divided into at least 2 parts: the training set and the test set.

Another way to split the dataset in search of more robust results is to separate it into 3 parts: training, testing, and validation/out of sample.

The purpose of this separation is to analyze whether the model has learned correctly from the training dataset or not.

The way this repository performs the data split is by seeking simplicity, allowing the user to define the dataset that the model has access to (`in-sample`), while the rest serves as the `out of sample` or validation dataset.

![data split](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/13677731-67aa-430f-93c6-777390c0a8e3)

This page is designed to be more responsive:

* The `Generate Parameters` button indicates which parameters have been updated.

![generate params](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/5091da1a-5ceb-421c-aa82-18dea015d4e4)

* The model creation can be canceled, and there is also information about what is being done.

![new_model_gif](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/7bf4ca68-9687-4bfb-bd0f-36c48cff4f69)