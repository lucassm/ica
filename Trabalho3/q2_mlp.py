#! coding: utf-8

u"""Questao 1 do terceiro trabalho de ICA.

Informacoes sobre o conjunto de dados:

The data are two strongly-blended Gaussians on a decaying
exponential baseline plus normally distributed zero-mean 
noise with variance = 6.25.

Data:          1 Response  (y)
               1 Predictor (x)
               250 Observations
               Average Level of Difficulty
               Generated Data

Model:         Exponential Class
               8 Parameters (b1 to b8)

               y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 )
                                   + b6*exp( -(x-b7)**2 / b8**2 ) + e
"""

from pylab import *
from tqdm import tqdm
from terminaltables import AsciiTable

# Definicao da Funcao Logistica e de sua derivada


def logistic(x):
    """Calcula a funcao logistica."""
    # a = 1.0 / (1.0 + exp(-x))
    # return a
    a = list()
    for i in x:
        if float(i) > 10.0:
            a.append(1.0)
        elif float(i) < -10.0:
            a.append(0.0)
        else:
            a.append(1.0 / (1.0 + exp(-float(i))))
    return array(a)


def logistic_diff(x):
    """Calcula a derivada da funcao logistica."""
    b = list()
    for i in x:
        if abs(float(i)) > 10.0:
            b.append(0.05)
        else:
            b.append(exp(float(i)) / (1.0 + exp(float(i)))**2 + 0.05)
    return array(b)

# Carrega os dados do conjunto de dados

f = open('Gauss3.dat.txt')
X = list()
Y = list()
for i, j in enumerate(f.readlines()):
    if i > 61:
        Y.append(float(j[0:11]))
        X.append(float(j[12:21]))
X = array(X)
Y = array(Y)

# X = arange(0, 10.0, 0.05)
# Y = sin(X)

X = reshape(X, (1, len(X)))
Y = reshape(Y, (1, len(Y)))

# pre-processamento dos dados

# Normalizacao componetes para media zero e variancia unitaria
for i in range(len(X)):
    mi = mean(X[i, :])  # media das linhas
    di = std(X[i, :])  # desvio-padrao das linhas
    X[i, :] = (X[i, :] - mi) / di

tx_acert_treino = list()
tx_acert_teste = list()
Tx_acert_teste = list()

# Definicao da Arquitetura da rede
n_ocultos = 40  # numero de neuronios na camada oculta
n_saida = 1  # numero de neuronios na camada de saida
n_entrada = 1  # numero de neuronios na camada de entrada
epocas = 1000  # numero de epocas de treinamento
txa = 0.005  # taxa de aprendizado da rede
mom = 0.001  # Fator de momento

nr = 1  # numero de rodadas de treinamento/teste

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

    # adicao do bias no vetor de entrada
    X_tr = vstack((-1.0 * ones((1, len(transpose(X_tr)))), X_tr))
    X_ts = vstack((-1.0 * ones((1, len(transpose(X_ts)))), X_ts))

    lX_tr, cX_tr = shape(X_tr)  # Tamanho da matriz de vetores de treinamento
    lX_ts, cX_ts = shape(X_ts)  # Tamanho da matriz de vetores de teste

    # inicializacao aleatoria do vetor de pesos
    W = 20.0 * rand(n_ocultos, lX_tr) - 10.0  # pesos dos neuronios da camada oculta
    M = 20.0 * rand(n_saida, n_ocultos + 1) - 10.0  # pesos dos neuronios da camada de saida

    W_old = W
    M_old = M
    # Fase de Treinamento da rede
    EQM = list()  # acumulador de erros por epoca

    # loop de epocas
    for j in tqdm(range(epocas), leave=True, ascii=True, ncols=100):

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
            x_tr = reshape(x_tr, (len(x_tr), 1))
            y_tr = reshape(y_tr, (len(y_tr), 1))
            # Camada Oculta
            U = dot(W, x_tr)
            U = reshape(U, (len(U), 1))

            Yi = logistic(U)  # saida entre 0 e 1, funcao logistica
            Yi = reshape(Yi, (len(Yi), 1))

            # Camada de Saida
            # adicao dos bias (-1) a entrada dos neuronios da camada de saida
            Y_ob = append(-1.0, Yi)
            Y_ob = reshape(Y_ob, (len(Y_ob), 1))

            Uk = dot(M, Y_ob)  # Ativacao dos neuronios da camada de saida
            Ok = Uk  # Funcao de Ativacao Linear

            # ---------------------------
            # Etapa de propagacao Reversa
            # ---------------------------

            # Calculo do Erro
            Ek = y_tr - Ok  # erro entre a saida desejada e saida obtida
            EQ += 0.5 * sum(Ek**2)  # soma do erro quadratico de todos os neuronios

            # Calculo dos Gradientes Locais
            Dk = Ok  # derivada da funcao de ativacao da camada de saida
            DDk = Ek  # * Dk  # gradiente local (camada de saida)

            Di = logistic_diff(Yi)  # derivada da funcao logistica da camada oculta
            Di = reshape(Di, (len(Di), 1))

            DDi = Di * dot(transpose(M[:, 1:]), DDk)  # gradiente local (camada oculta)


            # Ajustes dos pesos da camada de saida
            # DDk = reshape(DDk, (len(DDk), 1))
            # Ek = reshape(Ek, (len(Ek), 1))
            # Y_ob = reshape(Y_ob, (len(Y_ob), 1))

            M_old, M = M, M + txa * Ek * transpose(Y_ob) + mom * (M - M_old)

            # Ajuste dos pesos da camada oculta
            # DDi = reshape(DDi, (len(DDi), 1))
            # x_tr = reshape(x_tr, (len(x_tr), 1))

            W_old, W = W, W + txa * DDi * transpose(x_tr) + mom * (W - W_old)
            # raw_input()

            # print 'x_tr: ', x_tr
            # print 'Ek: ', Ek
            # print 'Ok: ', Ok
            # print 'y_tr', y_tr
            # print 'DDk: ', DDk
            # print 'DDi: ', DDi

            # raw_input()
        # Media do erro quadratico por epoca de treinamento
        EQM.append(EQ / cX_tr)

    plot(EQM)
    plt.show()
    # Etapa de Generalizacao
    EQ2 = 0.0
    Y2 = list()
    # loop de amostras de teste
    for x_ts in transpose(X):
        x_ts = vstack((-1.0 * ones((1, len(transpose(x_ts)))), x_ts))

        x_ts = reshape(x_ts, (len(x_ts), 1))

        # Camada Oculta
        Ui = dot(W, x_ts)  # Ativacao (net) dos neuronios da camada oculta
        Ui = reshape(Ui, (len(Ui), 1))

        Yi = logistic(Ui)   # Saida entre  0 e 1, funcao logistica

        # Camada de Saida
        # adicao dos bias (-1) a entrada dos neuronios da camada de saida
        Y_ob = append(-1.0, Yi)
        Y_ob = reshape(Y_ob, (len(Y_ob), 1))

        Uk = dot(M, Y_ob)  # Ativacao dos neuronios da camada de saida
        Ok = Uk  # Funcao de Ativacao Linear

        Y2.append(Ok)  # Armazena a saida da rede

    Y2 = array(Y2)
    # plot(Y2)
    # plt.show()
    # Calculo da Taxa de Acertos, comparando
    # a saida gerada pela rede e o vetor de saida
    # fornecido pelo conjunto de dados
    acertos = 0.0
    for y1, y2 in zip(Y2, transpose(Y_ts)):
        pass

# plot(X, Y, 'ro')
