## Objetivo 🎯
Essa página possui 3 objetivos:

1. Ser uma interface em que o usuário cria um modelo de Machine Learning de uma forma simplificada.
2. Permitir um alto grau de customização para o usuário com os menus: `Obter dados`, `Modificar Indicadores`, `Modificar Parâmetros de Features`, `Modificar Parâmetros do Modelo`, `Modificar a Divisão de Dados`, cada menu com uma responsabilidade especifica.
3. Obter as indicações do modelo.
 
 ## Dados 📊

O menu `Obter dados` Tem como objetivo permitir o usuário a escolher qual criptomoeda ele deseja que o modelo utilize os dados, como por exemplo o `Simbolo`, que deverá conter o nome do `par` que o modelo utilizará em sua base de dados, e deverá ser `exatamente` como é visto na corretora.

![[Pasted image 20240102181111.png]]

## Indicadores 💹

No menu `Modificar indicadores` permite selecionar qual indicador poderá ser utilizado, e caso haja alguma configuração adicional para esse indicador, será adicionado um menu para configuração:


Sem nenhuma configuração adicional:
![[Pasted image 20240102181632.png]]

Com alguma nova configuração adicional:
![[Pasted image 20240102181619.png]]
## Features ♟️
Em termos de trading, as features são metaforicamente o `setup` do modelo. Em outras palavras elas são conjunto de regras do modelo.

Nesse repositório o algoritmo utilizado é o `Gradient Boosting` da biblioteca `XGBoost`. 

Todas as features do modelo são features númericas, tendo isso em vista, os 3 inputs numéricos são os `Pontos de corte` que o modelo utilizará para definir a faixa de valor na qual será utilizado para a transformar cada feature numérica em três features binárias.

O menu  `Seleção de Features` permite ao usuário escolher qual feature binária será utilizada.

- a ordem selecionada dessas features pode impactar no resultado final do modelo.
![[Pasted image 20240102182616.png]]

## Parâmetros do modelo 🤖

Os parâmetros do modelo são valores que definem como o modelo irá aprender a classificar as features binárias, a cada nova configuração nos indicadores ou features é recomendado clicar no botão `Gerar Parâmetros` porque será buscado os melhores valores para o modelo.

Valores padrões:
![[Pasted image 20240102182753.png]]

Valores gerados:
![[Pasted image 20240102184008.png]]

## Divisão de Dados 📈📉📈

Quando se trata de Machine Learning a base de dados é no mínimo divida em 2 partes, a de treino, e a outra de treinamento.

Outra forma também de dividir essa base de dados em busca de resultados mais sólidos é separar em 3 partes, treinamento, teste, validação/fora da amostra

O objetivo dessa separação é permitir que seja possível analisar se o modelo aprendeu corretamente com a base de dados de treinamento ou não.

A forma que é feito a divisão nesse repositório foi buscando a simplicidade permitindo com que o usuário defina a base de dados que o modelo tem acesso (`in-sample`) e todo o restante ser a base de dados `out of sample` ou validação.

![[Pasted image 20240102183618.png]]


Essa página foi projetada para ser mais responsiva:

* o botão de gerar parâmetros indica quais parâmetros foram atualizados

![[gerarparams.gif]]

* A criação do modelo pode ser cancelada, e também há a informação do que está sendo feito.

![[createmodel.gif]]
