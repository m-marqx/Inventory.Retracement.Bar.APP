## Objetivo 🎯
Essa página possui 3 objetivos:

1. Ser uma interface em que o usuário cria um modelo de Machine Learning de uma forma simplificada.
2. Permitir um alto grau de customização para o usuário com os menus: `Obter dados`, `Modificar Indicadores`, `Modificar Parâmetros de Features`, `Modificar Parâmetros do Modelo`, `Modificar a Divisão de Dados`, cada menu com uma responsabilidade especifica.
3. Obter as indicações do modelo.
 
 ## Dados 📊

O menu `Obter dados` Tem como objetivo permitir o usuário a escolher qual criptomoeda ele deseja que o modelo utilize os dados, como por exemplo o `Simbolo`, que deverá conter o nome do `par` que o modelo utilizará em sua base de dados, e deverá ser `exatamente` como é visto na corretora.

![Get Data](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/ad3b1938-3b54-4cd5-89ad-980f3eb73430)


## Indicadores 💹

No menu `Modificar indicadores` permite selecionar qual indicador poderá ser utilizado, e caso haja alguma configuração adicional para esse indicador, será adicionado um menu para configuração:


Sem nenhuma configuração adicional:

![indicators modify1](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/74e9e6c2-db3e-4783-bcee-6afac7f1cae0)


Com alguma nova configuração adicional:

![indicators modify2](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/84b0ff4d-5448-4aec-8b92-38ad165cc8f7)

## Features ♟️
Em termos de trading, as features são metaforicamente o `setup` do modelo. Em outras palavras elas são conjunto de regras do modelo.

Nesse repositório o algoritmo utilizado é o `Gradient Boosting` da biblioteca `XGBoost`. 

Todas as features do modelo são features númericas, tendo isso em vista, os 3 inputs numéricos são os `Pontos de corte` que o modelo utilizará para definir a faixa de valor na qual será utilizado para a transformar cada feature numérica em três features binárias.

O menu  `Seleção de Features` permite ao usuário escolher qual feature binária será utilizada.

- a ordem selecionada dessas features pode impactar no resultado final do modelo.

![modify features](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/98e2ac70-e89e-48a3-90d5-64706ea00c36)

## Parâmetros do modelo 🤖

Os parâmetros do modelo são valores que definem como o modelo irá aprender a classificar as features binárias, a cada nova configuração nos indicadores ou features é recomendado clicar no botão `Gerar Parâmetros` porque será buscado os melhores valores para o modelo.

Valores padrões:

![default model params](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/9ac35d28-5afb-476e-9540-54762b881409)


Valores gerados:

![new model params](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/a860e2f1-05cf-4c15-b4ba-ca37b6ebd100)


## Divisão de Dados 📈📉📈

Quando se trata de Machine Learning a base de dados é no mínimo divida em 2 partes, a de treino, e a outra de treinamento.

Outra forma também de dividir essa base de dados em busca de resultados mais sólidos é separar em 3 partes, treinamento, teste, validação/fora da amostra

O objetivo dessa separação é permitir que seja possível analisar se o modelo aprendeu corretamente com a base de dados de treinamento ou não.

A forma que é feito a divisão nesse repositório foi buscando a simplicidade permitindo com que o usuário defina a base de dados que o modelo tem acesso (`in-sample`) e todo o restante ser a base de dados `out of sample` ou validação.

![data split](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/1b509efe-df29-4962-ad12-631de3dfcaa9)



Essa página foi projetada para ser mais responsiva:

* o botão de gerar parâmetros indica quais parâmetros foram atualizados

![generate params](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/d70da265-3d1f-47d9-a7e6-a89b6471e7cc)


* A criação do modelo pode ser cancelada, e também há a informação do que está sendo feito.

![new_model_gif](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/beedad9c-af59-4024-987f-e8379392b504)
