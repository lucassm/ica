#!coding:utf8

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm

# Evolucao Diferencial

# xi = target vector
# vi = mutant vector
# ui = trial vector

# carrega o conjunto de dados
aero = np.loadtxt('aerogerador.dat')

# funcao custo
def func_custo(ui):
    lambda_ = 1e3
    p = np.poly1d(ui)
    erro_total = 0
    for l in aero:
        e = l[1] - p(l[0])
        erro_total += e**2
    erro_total += lambda_ * np.linalg.norm(ui)**2
    return erro_total

# populacao inicial
num_pop = 10 # numero de individuos da populacao
pop = 4.0 * np.random.rand(num_pop, 5) - 2.0 # populacao gerada aleatoriamente

# parametros do algoritmo
F = 0.8 # constante de passo
CR = 0.6 # taxa de cruzamento

# numero de geracoes
num_gera = 50

erro = list()

# laco que da origem as 
# geracoes pre-determinadas nos
# parâmetros do algoritmo
for g in tqdm(range(num_gera), leave=True, ascii=True, ncols=100):
    # percorre cada um dos individuos
    # da populacao (target vectors) 
    # para a geracao atual
    for i, ind in enumerate(pop):

        # escolha aleatoria dos 3 vetores 
        # que dao origem ao vetor mutado
        numbers = range(num_pop)
        numbers.remove(i)
        r1, r2, r3 = sample(numbers, 3)

        # definicao do vetor mutado
        vi = pop[r1] + F * (pop[r2] - pop[r3]) # vetor mutado

        # geracao do trial vector
        ui = np.zeros(5)
        rand_indice = np.random.randint(5) # indice aleatorio de cruzamento obrigatorio

        # laco percorre cada um dos 
        # indices do individuo atual
        # para realizar cruzamento
        for k, j in enumerate(ind):

            # avalia qual dos indices serao cruzados
            # de acordo com a taxa de cruzamento CR
            if np.random.rand() <= CR or k == rand_indice:
                ui[k] = vi[k]
            else:
                ui[k] = j

        # avaliacao da funcao custo
        # do trial vector
        c1 = func_custo(ui)
        c2 = func_custo(ind)
        if c1 < c2:
            pop[i] = ui
            erro.append(c1)
        else:
            erro.append(c2)

# plt.plot(erro)
# plt.show()

x = np.arange(0, 15, 0.01)
menor_custo = np.inf

for i in pop:
    if func_custo(i) < menor_custo:
        p = np.poly1d(i)
        menor_custo = func_custo(i)


plt.plot(aero[:,0], aero[:,1], 'go')
plt.plot(x, p(x), lw=3.0)
plt.xlabel('velocidade do vento (m/s)')
plt.ylabel('potencia gerada (kW)')
plt.grid(True)
plt.legend(['Conj. de dados aerogerador',
           'Polinomino gerado via DE'])
plt.show()
