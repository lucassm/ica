#! coding:utf-8

"""Particle Swarm Optimization para minimizacao da funcao de 
Rastringin no intervalo x=[-5, 12] e y=[5, 12]"""

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm

def func_custo(x, y):
    """Calcula o valor da funcao a ser minimizada
    """
    z = 20.0 + x**2 + y**2 - 10.0 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return z

def aptidao_melhor_individuo(pop):
    """Encontra e retorna a melhor aptidao dos
    individuos da populacao
    """
    aptidao = np.inf
    for i in pop:
        custo = func_custo(i[0], i[1])
        if custo < aptidao:
            aptidao = custo

    return aptidao

def aptidao_media_populacao(pop):
    """Calcula e retorna a media da aptidao dos 
    individuos da populacao
    """
    ap = list()
    for i in pop:
        ap.append(func_custo(i[0], i[1]))
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
        custo = func_custo(i[0], i[1])
        if custo < aptidao:
            aptidao = custo
            melhor_ind = i

    return melhor_ind

################################
##  PARAMETROS DO ALGORITMO   ##
################################

# Coeficientes aceleradores
c1 = 2.05
c2 = 2.05

w = 0.4 #  parametro de inercia

num_pop = 60 # numero de individuos da populacao
num_gera = 6000 # numero de geracoes

# populacao gerada aleatoriamente
pop_x = 17 * np.random.rand(num_pop) - 5
pop_y = 7 * np.random.rand(num_pop) + 5

xi = np.column_stack((pop_x, pop_y))
vi = pop_x = 12 * np.random.rand(num_pop, 2) - 12

# pi e inicializado com a populacao inicial
pi = xi
menor_custo = np.inf

# para definir pg e necessário encontrar a particula com
# o menor valor de funcao custo
for i in xi:
    if func_custo(i[0], i[1]) < menor_custo:
        pg = i
        menor_custo = func_custo(i[0], i[1])

# Dados a serem coletados no decorrer das geracoes
apt_melhor = list()
apt_media = list()
erro = list()

##################################
## EXECUCAO DO LACO DE GERACOES ##
##################################
for g in tqdm(range(num_gera), leave=True, ascii=True, ncols=100):

    # For precorre cada uma das particulas
    # para atualizar suas respectivas velocidades
    # e posicoes 
    for e, i in enumerate(xi):
        
        # Gera números pseudo-aleat. uniform. distrib.
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        # Atualizacoes
        vi[e] = w * vi[e] + c1 * r1 * (pi[e] - i) + c2 * r2 * (pg - i)
        i = i + vi[e]

        # restricao do problema
        if i[0] > 12 or i[0] < -5:
            break

        if i[1] > 12 or i[1] < 5:
            break 

        # Calculo da funcao custo e atualizacao
        # dos menores valores globais e individuais
        p1 = func_custo(i[0], i[1])
        p2 = func_custo(pi[e][0], pi[e][1])
        if p1 < p2:
            pi[e] = i
            p3 = func_custo(pg[0], pg[1])
            # atualizacao de minimo local
            if p1 < p3:
                # atualizacao de minimo global
                pg = i

    # Armazenamento de informacoes sobre a populacao
    # no decorrer das geracoes 
    apt_media.append(aptidao_media_populacao(xi))
    apt_melhor.append(aptidao_melhor_individuo(xi))

##########################################
## Visualizacao dos resultados          ##
##########################################
plt.plot(apt_media, 'r--', lw=3.0)
plt.plot(apt_melhor, 'g--', lw=3.0)
plt.grid(True)
plt.legend(['aptidao media da populacao',
            'aptidao do melhor individuo da populacao'])
plt.show()
