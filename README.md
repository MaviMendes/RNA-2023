# Trabalho final da disciplina de Redes neurais artificiais - DCM USP RP - 2023

## Dataset

Dados referentes a hábitos financeiros de pessoas. Coleta feita de forma anônima via Google Forms

## Organização dos arquivos

data_processing, avalatiation e plotting contém métodos auxiliares para, respectivamente:
* Processar o dataset, transformando colunas string em valores inteiros
* Separar o dataset entre x e y, sendo x as colunas de parêmtros e y a coluna com as classes

holdout e cross_validation utilizam avaliation e plotting para, em seus respectivos datasets, criar mlp, ver o score para cada função de ativação e resolvedor, e plotar a curva de perda por iteração, bem como a matriz de confusão

Ao final da execução de holdout e cross_validation, podemos avaliar o desempenho dos diferentes MLPs criados, considerando as variações de activation function, solver e método de separação dos dados

Quantidade de pessoas em cada classe:

Responsável: 27
Irresponsável: 26
Intermediário: 21

