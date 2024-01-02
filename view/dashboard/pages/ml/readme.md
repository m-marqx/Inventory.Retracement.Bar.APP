## PT-BR
Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/Hoffman-IRB/blob/master/view/dashboard/pages/ml/Readme%20-%20BR.md)

## Objective 🎯

This page has 3 objectives:

1. To be an interface where the user creates a Machine Learning model in a simplified way.
2. To allow a high degree of customization for the user with the menus: `Get Data`, `Modify Indicators`, `Modify Feature Parameters`, `Modify Model Parameters`, `Modify Data Split`, each menu with a specific responsibility.
3. To obtain model predictions.

## Data 📊

The `Get Data` menu aims to allow the user to choose which cryptocurrency the model will use for data, such as the `Symbol`, which should contain the name of the `pair` that the model will use in its database, and should be `exactly` as seen on the exchange.  

![[Pasted image 20240102191253.png]]

## Indicators 💹

The `Modify Indicators` menu allows selecting which indicator can be used, and if there is any additional configuration for that indicator, a menu for configuration will be added:

Without any additional configuration: 

![[Pasted image 20240102191317.png]]

With some additional configuration:

![[Pasted image 20240102191809.png]]

## Features ♟️

In terms of trading, features are metaphorically the model's `setup`. In other words, they are a set of rules for the model.

In this repository, the algorithm used is the `Gradient Boosting` from the `XGBoost` library.

All model features are numerical features. With this in mind, the 3 numerical inputs are the `Threshold Points` that the model will use to define the value range that will be used to transform each numerical feature into three binary features.

The `Feature Selection` menu allows the user to choose which binary feature will be used.
 
- The selected order of these features can impact the final result of the model. 

![[Pasted image 20240102191753.png]]
## Model Parameters 🤖

Model parameters are values that define how the model will learn to classify binary features. Whenever there's a new configuration in the indicators or features, it's recommended to click the `Generate Parameters` button to search for the best values for the model.

Default values:

![[Pasted image 20240102191728.png]]

Generated values:

![[Pasted image 20240102191622.png]]

## Data Split 📈📉📈

When it comes to Machine Learning, the dataset is typically divided into at least 2 parts: the training set and the test set.

Another way to split the dataset in search of more robust results is to separate it into 3 parts: training, testing, and validation/out of sample.

The purpose of this separation is to analyze whether the model has learned correctly from the training dataset or not.

The way this repository performs the data split is by seeking simplicity, allowing the user to define the dataset that the model has access to (`in-sample`), while the rest serves as the `out of sample` or validation dataset.

![[Pasted image 20240102191700.png]]


This page is designed to be more responsive:  

* The `Generate Parameters` button indicates which parameters have been updated.

![[generate_params_en.gif]]

* The model creation can be canceled, and there is also information about what is being done.

![[createmodel-en.gif]]!