## Ideia Inicial

O repositório tem objetivo criar um modelo que coleta os dados da Binance, a maior exchange internacional, usando a biblioteca `python-binance`, após isso os dados são manipulados para obter os resultados da estratégia `Inventory Retracement Bar`, desenvolvida por `Rob Hoffman`. A estratégia se mostrava promissora, já que o autor alega que ela trouxe vitórias em várias competições de trading, porém com as configurações padrões do setup foi observado que os resultados insatisfatórios no mercado de criptomoedas.

## Objetivo

Criar um modelo em que o sinal ocorre quando a sombra do candle é igual ou superior a 45% do corpo do mesmo. 

![Signal](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/3e1f7810-19f6-45b3-ba9f-b7a940463c77)

## Sinal de entrada

Por padrão a Média Movel Simples de 20 períodos é utilizada para indicar a tendência, embora o autor sugira que seja buscado trades com ela possuindo uma inclinação de 45 graus, essa ideia foi descartada devido à subjetividade do ângulo quando se varia o zoom do gráfico. 

![62degree](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/595ce7f2-3440-49b8-933e-78ad29c953f7) 

![45degree](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/946d999d-75a1-4038-8561-5c05145dc7f6)

## Trading

A entrada ocorre quando o candle rompe o valor máximo do candle que gerou o sinal, apesar de reduzir o número total de entradas, isso não é um problema devido à alta quantidade de sinais no setup. 

![start](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/9b93b458-6a31-4a55-a7e0-f12945c5cd33)

O valor do alvo depende do `payoff` escolhido, sendo este a amplitude do candle de sinal, e o valor do stop deve ser na mínima do candle de sinal ou no menor valor dos últimos N candles. O exemplo segue com o `Payoff` de 2 e o stop na mínima do candle que gerou o sinal. 

![end](https://github.com/m-marqx/Hoffman-IRB/assets/124513922/071a7775-6e59-4ae8-b9a1-b110559da699)

## Como utilizar
 
Para melhorar a visualização dos resultados, foi criado um Dashboard utilizando o `Dash`, da mesma equipe por trás do `Plotly`. O Dashboard possui 2 páginas: `homepage` e `backtest`, cada uma com seu objetivo específico, com ambas buscando proporcionar aos usuários uma customização simples e fácil para visualizar resultados com diferentes valores nos parâmetros da estratégia.

Acesse [homepage](https://github.com/m-marqx/Hoffman-IRB/blob/master/view/dashboard/pages/home/readme%20-%20BR.md) ou [backtest](https://github.com/m-marqx/Hoffman-IRB/blob/master/view/dashboard/pages/backtest/readme%20-%20BR.md) para informações detalhadas sobre a utilização de cada página do Dashboard.
