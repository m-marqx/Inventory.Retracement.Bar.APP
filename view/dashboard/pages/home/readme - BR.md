## Objetivo 🎯
A homepage possui 2 objetivos:

1. Ser uma interface em que o usuário customiza e visualiza de forma simples o resultado da estratégia IRB
2. Permitir um alto grau de customização para o usuário a página possui 5 menus: `Obter dados`, `Modificar Parâmetros de indicadores`, `Modificar Parâmetros da estratégia`, `Modificar Parâmetros da Têndencia`, `Modificar Parâmetros de Resultado`, cada menu com uma responsabilidade especifica.
 
 ## Dados 📊

O menu `Obter dados` Tem como objetivo permitir o usuário a escolher em qual tipo de mercado ele deseja obter os dados:
 
`Spot (Mercado à vista), Futuros, Preço de Marcação (Mark Price)`: ele deverá selecionar o Timeframe e o Símbolo que deseja obter os dados

![image](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/54cdef84-eaae-4498-b412-5b1abc6924b5)

`Outro`: Caso o usuário já tenha os dados ele poderá selecionar a opção `Outro` que permite arquivos no formato `parquet` e `csv`.

![](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/b1e560c9-93a0-4bee-b663-6d36c755ca80)

## Indicadores 💹

No menu `Modificar Parâmetros de indicadores` permite o usuário customizar os indicadores EMA, MACD, e CCI

![](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/8429c69b-9d5b-4347-b158-073efcd2dc90)

## Estratégia ♟️

No menu `Modificar Parâmetros da estratégia` permite ao usuário ajustar os parametros de sinal, mas também de entrada e saida do setup, (Possivelmente o tamanho do tick em um update futuro será removido)

![](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/dfe1e141-8db1-4726-816c-239b99025d8f)

`Minima mais baixa`: define onde será colocado a mínima, como padrão é na minima do candle atual.

`Tamanho do Tick`: define o valor que será adicionado no valor da entrada ou reduzido no valor do stop

`Payoff`: define onde será colocado o alvo, como padrão é `2` vezes o candle de sinal

`Porcentagem do Wick`: define qual é a % minima aceita para o candle de sinal

## Definição de tendência 📈

No menu `Modificar Parâmetros da Têndencia` serve para o usuário selecionar como a tendência será definida pelos indicadores, e quais indicadores serão utilizados e qual o tipo de preço (Abertura, Máxima, Minima, Fechamento) será utilizada para identificar a tendência de alta.

![](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/2fa0f41d-9dc2-40d8-9afe-36fb7be530fd)


## Resultado 🪙

O menu `Modificar Parâmetros de Resultado` tem como objetivo modificar os parâmetros para o gerenciamento de risco, enquanto o sub-menu `Modificar Configurações de Resultado`, para definir como serão mostrados os resultados e qual é o ativo em que o usuário quer simular que o retorno, se é em USD, ou é na moeda operada

![](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/9ca3f493-5759-464a-903d-7af6f7f7643c)


