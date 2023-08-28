## Objetivo 🎯

Essa página tem como objetivo principal ser uma interface de simples customização para a visualização de um backtest em que é utilizado o GridSearch (também conhecido como força bruta) dos resultados, para identificar dois principais pontos:

1. Quais são os parametros que fazem a estratégia ser Lucrativa
2. Quais são os **melhores valores** de parametros para obter lucro nessa estratégia

Esses dois pontos ajudam a identificar qual é a faixa ideal de valores para ter bons resultados (teóricos) com a estratégia IRB.

Semelhante a Homepage essa página tem também como objetivo possuir um alto grau de customização e por conta disso possui quase todos os menus da homepage sendo eles: `Obter dados`, `Modificar Parâmetros de indicadores`, `Modificar Parâmetros da estratégia`, `Modificar Parâmetros de Resultado`, e com a adição do menu `Modificar Parâmetros de Hardware`.

## Dados 📊

O menu `Obter dados` Tem como objetivo permitir o usuário a escolher em qual tipo de mercado ele deseja obter os dados:
 
`Spot (Mercado à vista), Futuros, Preço de Marcação (Mark Price)`: ele deverá selecionar o Timeframe e o Símbolo que deseja obter os dados

![image](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/b936bb2e-f9ef-4bb5-960c-520a073a2eca)

`Outro`: Caso o usuário já tenha os dados ele poderá selecionar a opção `Outro` que permite arquivos no formato `parquet` e `csv`.

![](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/b1e560c9-93a0-4bee-b663-6d36c755ca80)

## Indicadores 💹

No menu `Modificar Parâmetros de Indicadores` apenas possui o indicador EMA para ser testado, será realizado um Grid Search com todas as possibilidades considerando todos os valores do valor do `Periodo Mínimo da EMA` até o `Periodo Máximo da EMA`.

![Indicators](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/18d81f59-38b8-4c08-9618-b11f8030549e)
## Estratégia ♟️

O menu `Modificar Parâmetros da Estratégia` serve para definir quais serão os valores mínimos e máximos da `Minima Mais Baixa`, `Payoff`, `Porcentagem do Wick` e adicionar todas as possibilidades possíveis no Grid Search

![Strategy](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/c869c491-a51b-4be6-a7e4-53e454e14c73)



## Resultado 🪙

O menu `Modificar Parâmetros de Resultado` tem como objetivo modificar os parâmetros para o gerenciamento de risco podendo selecionar multiplas opções, enquanto o sub-menu `Modificar Configurações de Resultado` serve para definir como serão mostrados os resultados e qual é o ativo em que o usuário quer simular que o retorno, se é em USD, ou é na moeda operada, como a tendência é criar um gráfico de dificil visualização devido alta quantidade de linhas mostradas no gráfico, o usuário pode definir se deseja apenas mostrar os resultados que terminaram positivos ou todos, incluindo os que terminaram no prejuízo

![Result](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/3fe0b7f5-9ade-4fcb-895a-832271eedb30)


## Hardware

O menu `Modificar Parâmetros de Hardware` tem como principal objetivo o usuário modificar a quantidade de Cores do CPU ou utilizar a GPU para o processamento do backtest, para evitar que o usuário utilize a opção da GPU sem as configurações necessárias ou sem ter uma, foi utilizado o método `.cuda.device_count()` do PyTorch. Caso ele possua será possível modificar as opções `Número de GPU` e nem `Número de workers da GPU`, e tão pouco irá poder selecionar a opção `GPU`

GPU não encontrada/configurada:

![GPU not found](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/320255a6-fe6b-4ae7-9876-631c358ec4be)

GPU encontrada e configurada:

![GPU found](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/866db730-4507-4fd8-95b8-403d83165d37)
