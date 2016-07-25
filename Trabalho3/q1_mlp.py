#! coding: utf-8

u"""Questao 1 do terceiro trabalho de ICA.

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

A ideia aqui e implementar uma rede neural do tipo Perceptron de Multiplas
Camadas com algoritmo de treinamento backpropagation.
"""

from pylab import *
from tqdm import tqdm
from terminaltables import AsciiTable


def logistic(x):
    """Calcula a funcao logistica."""
    return 1.0 / (1.0 + exp(-x))


def logistic_diff(x):
    """Calcula a derivada da funcao logistica."""
    return exp(-x) / (1.0 + exp(-x))**2


X = loadtxt('pacientes.txt')

Y = loadtxt('patologias.txt')

# pre-processamento dos dados

# Normalizacao componetes para media zero e variancia unitaria
for i in range(len(X)):
    mi = mean(X[i, :])  # media das linhas
    di = std(X[i, :])  # desvio-padrao das linhas
    X[i, :] = (X[i, :] - mi) / di

tx_acert_treino = list()
tx_acert_teste = list()
Tx_acert_teste = list()

nr = 100  # numero de rodadas de treinamento/teste

# Realizacao de Rodadas de Treinamento/Teste da rede
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

    # Definicao da Arquitetura da rede
    n_ocultos = 10  # numero de neuronios na camada oculta
    n_saida = 6  # numero de neuronios na camada de saida
    epocas = 100  # numero de epocas de treinamento
    txa = 0.1  # taxa de aprendizado da rede

    # adicao do bias no vetor de entrada
    X_tr = vstack((-1.0 * ones((1, len(transpose(X_tr)))), X_tr))
    X_ts = vstack((-1.0 * ones((1, len(transpose(X_ts)))), X_ts))

    lX_tr, cX_tr = shape(X_tr)  # Tamanho da matriz de vetores de treinamento
    lX_ts, cX_ts = shape(X_ts)  # Tamanho da matriz de vetores de teste

    # inicializacao aleatoria do vetor de pesos
    W = rand(n_ocultos, lX_tr)  # pesos dos neuronios da camada oculta
    M = rand(n_saida, n_ocultos + 1)  # pesos dos neuronios da camada de saida

    # Fase de Treinamento da rede
    EQM = list()  # acumulador de erros por epoca

    # loop de epocas
    for j in range(epocas):

        EQ = 0.0  # Erro por epoca

        # gera indices aleatoriamente
        I = permutation(len(transpose(X_tr)))

        # randomizacao dos vetores de treinamento
        X_tr, Y_tr = X_tr[:, I], Y_tr[:, I]

        # loop de amostras de treinamento
        for x_tr, y_tr in zip(transpose(X_tr), transpose(Y_tr)):
            # --------------------------
            # Etapa de propagacao Direta
            # --------------------------

            # Camada Oculta
            U = dot(W, x_tr)
            Yi = logistic(U)  # saida entre 0 e 1, funcao logistica

            # Camada de Saida
            # adicao dos bias (-1) a entrada dos neuronios da camada de saida
            Y_ob = append(-1.0, Yi)
            Uk = dot(M, Y_ob)  # Ativacao dos neuronios da camada de saida
            Ok = logistic(Uk)  # Saida entre 0 e 1, funcao logistica

            # ---------------------------
            # Etapa de propagacao Reversa
            # ---------------------------

            # Calculo do Erro
            Ek = y_tr - Ok  # erro entre a saida desejada e saida obtida
            EQ += 0.5 * sum(Ek**2)  # soma do erro quadrat. de tods os neur.

            # Calculo dos Gradientes Locais
            Dk = logistic_diff(Ok)  # derivada da func. logist. da cam. de said
            DDk = Ek * Dk  # gradiente local (camada de saida)

            Di = logistic_diff(Yi)  # derivada da duncao logist. da cam. ocult
            DDi = Di * dot(transpose(M[:, 1:]), DDk)  # grad. loc. (cam. ocult)

            # Ajustes dos pesos da camada de saida
            DDk = reshape(DDk, (len(DDk), 1))
            Y_ob = reshape(Y_ob, (len(Y_ob), 1))

            M += txa * DDk * transpose(Y_ob)

            # Ajuste dos pesos da camada oculta
            DDi = reshape(DDi, (len(DDi), 1))
            x_tr = reshape(x_tr, (len(x_tr), 1))

            W += txa * DDi * transpose(x_tr)

        # Media do erro quadratico por epoca de treinamento
        EQM.append(EQ / cX_tr)

    # Etapa de Generalizacao
    EQ2 = 0.0
    OUT2 = list()
    # loop de amostras de teste
    for x_ts in transpose(X_ts):

        # Camada Oculta
        Ui = dot(W, x_ts)  # Ativacao (net) dos neuronios da camada oculta
        Yi = logistic(Ui)   # Saida entre  0 e 1, funcao logistica

        # Camada de Saida
        # adicao dos bias (-1) a entrada dos neuronios da camada de saida
        Y_ob = append(-1.0, Yi)
        Uk = dot(M, Y_ob)  # Ativacao dos neuronios da camada de saida
        Ok = logistic(Uk)  # Saida entre 0 e 1, funcao logistica

        OUT2.append(Ok)  # Armazena a saida da rede

    # Converte o maior elemento da coluna da matriz
    # em 1 e insere 0 nos demais
    Y2 = list()  # Matriz de saida padronizada em 0s e 1s
    for u in OUT2:
        a = argmax(u)
        yn = zeros(shape=len(u))
        yn[a] = 1.0
        Y2.append(yn)
    Y2 = transpose(array(Y2))

    # Calculo da Taxa de Acertos, comparando
    # a saida gerada pela rede e o vetor de saida
    # fornecido pelo conjunto de dados
    acertos = 0.0
    for y1, y2 in zip(transpose(Y2), transpose(Y_ts)):
        if (y1 == y2).all():
            acertos += 1.0
    tx_acert_teste.append(acertos / len(transpose(Y_ts)))

# Estatisticas Descritivas
Media = mean(tx_acert_teste)
Mediana = median(tx_acert_teste)
Maxima = max(tx_acert_teste)
Minima = min(tx_acert_teste)
DevPadrao = std(tx_acert_teste)

# Exibicao dos resultados das Estatisticas em formato de tabela

table_data = [['Media', 'Mediana', 'Max', 'Min', 'Dev. Padrao'],
              [str(round(Media, 2)),
               str(round(Mediana, 2)),
               str(round(Maxima, 2)),
               str(round(Minima, 2)),
               str(round(DevPadrao, 2))]]
tab = AsciiTable(table_data)

print '\n\n', tab.table
