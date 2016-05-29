#! coding:utf-8

"""Evolucao Diferencial para minimizacao da funcao de 
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

F = 0.8  # constante de passo
CR = 0.6  # taxa de cruzamento
num_pop = 50  # numero de individuos da populacao
num_gera = 5000  # numero de geracoes


# populacao gerada aleatoriamente
pop_x = 17 * np.random.rand(num_pop) - 5
pop_y = 7 * np.random.rand(num_pop) + 5

pop = np.column_stack((pop_x, pop_y))

# Dados a serem coletados no decorrer das geracoes
apt_melhor = list()
apt_media = list()
erro = list()

##################################
## EXECUCAO DO LACO DE GERACOES ##
##################################
for g in tqdm(range(num_gera), leave=True, ascii=True, ncols=100):
    
    # For percorre todos os individuos da populacao
    for i, ind in enumerate(pop):

        # Escolha dos tres vetores que dao origem ao vetor
        # mutado
        numbers = range(num_pop)
        numbers.remove(i)
        r1, r2, r3 = sample(numbers, 3)

        # Definicao do Vetor Mutado
        vi = pop[r1] + F * (pop[r2] - pop[r3]) # vetor mutado

        # Geracao do Trial Vector
        ui = np.zeros(2)
        rand_indice = np.random.randint(2) # indice aleat. de cruz. obrigat.

        # For percorre cada um dos 
        # indices do individuo atual
        # para realizar CRUZAMENTO
        for k, j in enumerate(ind):

            # Avalia quais indices serao cruzados
            # de acordo com a taxa de cruzamento CR
            if np.random.rand() <= CR or k == rand_indice:
                ui[k] = vi[k]
            else:
                ui[k] = j

        # Aplica a restricao do problema
        if ui[0] > 12 or ui[0] < -5:
            break

        if ui[1] > 12 or ui[1] < 5:
            break 

        # Avalia a funcao custo do Trial Vector
        c1 = func_custo(ui[0], ui[1])
        c2 = func_custo(ind[0], ind[1])
        
        if c1 < c2:
            pop[i] = ui
            erro.append(c1)
        else:
            erro.append(c2)

    # Armazenamento de informacoes sobre a populacao
    # no decorrer das geracoes 
    apt_media.append(aptidao_media_populacao(pop))
    apt_melhor.append(aptidao_melhor_individuo(pop))

##########################################
## Visualizacao dos resultados          ##
##########################################
plt.plot(apt_media, 'r--', lw=3.0)
plt.plot(apt_melhor, 'g--', lw=3.0)
plt.grid(True)
plt.legend(['aptidao media da populacao', 
            'aptidao do melhor individuo da populacao'])
plt.show()
