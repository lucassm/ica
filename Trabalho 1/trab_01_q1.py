#! coding:utf-8

"""Algoritmo Genetico para minimizacao da funcao de 
Rastringin no intervalo x=[-5, 12] e y=[5, 12]"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randint, random_integers
from tqdm import tqdm

#####################################
## METODOS UTILIZADOS NO ALGORITMO ##
#####################################

def func_custo(x, y):
    """Calcula o valor da funcao a ser minimizada
    """
    z = 20.0 + x**2 + y**2 - 10.0 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return z

def calc_aptidao(pop):
    """Calcula a aptidao de cada um dos individuos
    da populacao e retorna a probabilidade de cada
    um ser escolhido para cruzamento de acordo com
    sua aptidao
    """
    ap = list()
    # representacao decimal dos cromossomos
    for i in pop:
        ap.append(
            func_custo(decode(i[0], -5, 12), decode(i[1], 5, 12))
            )

    # valores de probabilidade obtidos de acordo com a aptidao
    prob = [round(j / sum(ap), 4) * 100.0 for j in ap]

    return prob


def bin2dec(indiv):
    """Decodifica um numero de binario para decimal
    com uma codificacao de 22 bits
    """
    return sum([a * 2**b for a, b in zip(reversed(indiv), range(22))])

def decode(indiv, min, max):
    """Decodifica o cromossomo do indivivuo em binario
    para um valor decimal dentro da escala min, max
    estabelecida
    """
    return min + (max - min) * bin2dec(indiv) / (2**22 - 1.0)

def torneio(pop, prob):
    """Implementa a selecao de dois individuos
    pelo metodo do torneio
    """
    a1 = np.random.randint(0, len(pop) - 1)
    a2 = np.random.randint(0, len(pop) - 1)
    if prob[a1] > prob[a2]:
        return pop[a2]
    else:
        return pop[a1]

def cross_over_uniform(p1, p2):
    """Iplementa a aplicacao de cross-over uniform
    no individuo de duas dimensoes
    """
    l = len(p1[0])
    masc = random_integers(0, 1, l)
    f1 = np.zeros((2, l))
    f2 = np.zeros((2, l))
    for i, j in enumerate(masc):
        if j == 1:
            f1[0][i] = p1[0][i]
            f1[1][i] = p1[1][i]
            
            f2[0][i] = p2[0][i]
            f2[1][i] = p2[1][i]
        else:
            f1[0][i] = p2[0][i]
            f1[1][i] = p2[1][i]

            f2[0][i] = p1[0][i]
            f2[1][i] = p1[1][i]
    return f1, f2

def aptidao_melhor_individuo(pop):
    """Encontra e retorna a melhor aptidao dos
    individuos da populacao
    """
    aptidao = np.inf
    for i in pop:
        custo = func_custo(decode(i[0], -5, 12), decode(i[1], 5, 12))
        if custo < aptidao:
            aptidao = custo

    return aptidao

def aptidao_media_populacao(pop):
    """Calcula e retorna a media da aptidao dos 
    individuos da populacao
    """
    ap = list()
    for i in pop:
        ap.append(
            func_custo(decode(i[0], -5, 12), decode(i[1], 5, 12))
            )
    media = sum(ap) / len(ap)
    return media

def melhor_individuo(pop):
    """Encontra e retorna o individuo com a 
    melhor aptidao entre os individuos
    da populacao
    """
    aptidao = np.inf
    melhor_ind = None
    for i in pop:
        custo = func_custo(decode(i[0], -5, 12), decode(i[1], 5, 12))
        if custo < aptidao:
            aptidao = custo
            melhor_ind = i

    return decode(melhor_ind[0], -5, 12), decode(melhor_ind[1], 5, 12)


################################
##  PARAMETROS DO ALGORITMO   ##
################################

num_pop = 50  # numero da populacao
num_gera = 200  # numero de geracoes a serem executadas
CR = 0.2  # taxa de cross-over
MR = 0.02  # taxa de mutacao


# Gera a polulacao aleatoriamente
pop = np.random.random_integers(0, 1, (num_pop, 2, 22))

# Dados a serem coletados no decorrer das geracoes
apt_melhor = list()
apt_media = list()


##################################
## EXECUCAO DO LACO DE GERACOES ##
##################################
for g in tqdm(range(num_gera), leave=True, ascii=True, ncols=100):
    # sleep(0.01)
    pop_ = list()

    # calculo das aptidoes
    prob = calc_aptidao(pop)
    
    # For percorre os individuos
    # da populacao em pares
    for i in range(num_pop/2):
        # Selecao
        p1 = torneio(pop, prob)
        p2 = torneio(pop, prob)

        # Cross-over
        if rand() < CR:
            # Realiza cross-over uniforme
            # e gera novos individuos
            f1,f2 = cross_over_uniform(p1, p2)
            pop_.append(f1)
            pop_.append(f2)
        else:
            # Repete os pais
            pop_.append(p1)
            pop_.append(p2)
    pop_ = np.array(pop_)

    # Mutacao
    for i in pop_:  # for percorre os genes do individuo
        for k, j in enumerate(i[0]):
            if rand() < MR:
                i[0][k] = int(not i[0][k])

        for k, j in enumerate(i[1]):
            if rand() < MR:
                i[1][k] = int(not i[1][k])
    pop = pop_
    
    # Armazenamento de informacoes sobre a populacao
    # no decorrer das geracoes 
    apt_media.append(aptidao_media_populacao(pop))
    apt_melhor.append(aptidao_melhor_individuo(pop))

##########################################
## Visualizacao dos resultados          ##
##########################################

plt.plot(apt_media, 'ro--')
plt.plot(apt_melhor, 'go--')
plt.grid(True)
plt.legend(['aptidao media da populacao', 
            'aptidao do melhor individuo da populacao'])
plt.show()