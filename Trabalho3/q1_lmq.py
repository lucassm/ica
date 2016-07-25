#! coding: utf-8
""" Questao 1 do terceiro trabalho de ICA.

informacoes sobre o conjunto de dados:

arquivo pacientes.txt: contem uma matriz de 34 linhas e 358 colunas, onde
cada coluna representa um paciente e cada linha representa uma caracteristica
do paciente.

arquivo patologias.txt: contem uma matriz de 6 linhas e 358 colunas, onde
cada coluna representa um paciente e cada coluna representa uma das doencas,
a serem detectadas, a doenca a que o pacinte pertencer, será marcado um 1, as
demais linhas serão preenchidas com 0.

A codificacao das patologias segue a seguinte identificacao:

[1,0,0,0,0,0] -> psoriase
[0,1,0,0,0,0] -> Derm. Seborreia
[0,0,1,0,0,0] -> Liquen Plano
[0,0,0,1,0,0] -> Pitiriase Rosea
[0,0,0,0,1,0] -> Derm. Cronica
[0,0,0,0,0,1] -> Pitiriase Rubra Pilar

A ideia e implementar um classificador linear de patologias por meio 
da seguinte expressao y = Ax, em que y e uma codificacao da patologia
x e um vetor com as caracteristicas do paciente e A e o classificador
propriamente dito.

E possivel obter tal classificador, da seguinte forma:

Y = A.X
Y.X' = A.(X.X')
Y.X'.inv(X.X') = A. (X.X').inv(X.X')
Y.X'.inv(X.X') = A
A = Y.X'.inv(X.X')

utilizando a matriz para obter classificacoes da seguinte forma:

yi = A.xi
Em que xi e um vetor de caracteristicas qualquer e yi e a resposta do
classificador
"""

from pylab import *
from tqdm import tqdm
from terminaltables import AsciiTable

X = loadtxt('pacientes.txt')

Y = loadtxt('patologias.txt')

nr = 100  # numero de rodadas de treinamento/teste

taxa_acertos = list()

# Realizacao de Rodadas de Treinamento/Teste
for i in tqdm(range(nr), leave=True, ascii=True, ncols=100):
    # gera indices aleatoriamente
    I = permutation(len(transpose(X)))

    # Define tamanho dos conjuntos de treinamento/teste
    ptr = 0.8  # Porcentagem usada para treino
    pts = 1.0 - ptr  # Porcentagem usada para teste

    n_tr = int(floor(ptr * len(transpose(X))))
    n_ts = int(ceil(pts * len(transpose(X))))

    # declaracao dos vetores de teste e treinamento
    X_tr, Y_tr = X[:, I[:n_tr]], Y[:, I[:n_tr]]
    X_ts, Y_ts = X[:, I[n_tr:]], Y[:, I[n_tr:]]

    # Fase de Treinamento do Classificador
    X_trt = transpose(X_tr)
    P1, P2 = dot(Y_tr, X_trt), inv(dot(X_tr, X_trt))
    A = dot(P1, P2)

    # Fase de Teste do Classificador

    Y_ob = dot(A, X_ts)
    Y_aux = list()
    for y in transpose(Y_ob):
        a = argmax(y)
        yn = zeros(shape=len(y))
        yn[a] = 1.0
        Y_aux.append(yn)
    Y_ob = transpose(array(Y_aux))

    acertos = 0.0
    for y1, y2 in zip(transpose(Y_ob), transpose(Y_ts)):
        if (y1 == y2).all():
            acertos += 1.0
    taxa_acertos.append(acertos / len(transpose(Y_ts)))

# Estatisticas Descritivas
Media = mean(taxa_acertos)
Mediana = median(taxa_acertos)
Maxima = max(taxa_acertos)
Minima = min(taxa_acertos)
DevPadrao = std(taxa_acertos)

# Exibicao dos resultados das Estatisticas em formato de tabela

table_data = [['Media', 'Mediana', 'Max', 'Min', 'Dev. Padrao'],
              [str(round(Media, 2)),
               str(round(Mediana, 2)),
               str(round(Maxima, 2)),
               str(round(Minima, 2)),
               str(round(DevPadrao, 2))]]
tab = AsciiTable(table_data)

print '\n\n', tab.table
