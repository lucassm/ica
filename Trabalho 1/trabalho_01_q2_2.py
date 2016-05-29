#!coding:utf8

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm

# Particle Swarm Optimization

# inicialização dos parâmetros

w = 0.5 # parametro de inercia

# coeficientes aceleradores
c1 = 2.05
c2 = 2.05

# carrega o conjunto de dados
aero = np.loadtxt('aerogerador.dat')

# funcao custo
def func_custo(ui):
    p = np.poly1d(ui)
    erro_total = 0
    for l in aero:
        e = l[1] - p(l[0])
        erro_total += e**2
    return erro_total


# populacao inicial
num_pop = 30 # numero de individuos da populacao

xi = 4.0 * np.random.rand(num_pop, 5) - 2.0 # posiçoes geradas aleatoriamente
vi = 4.0 * np.random.rand(num_pop, 5) - 2.0 # velocidades geradas aleatoriamente

pi = xi
menor_custo = np.inf

for i in xi:
    if func_custo(i) < menor_custo:
        pg = i
        menor_custo = func_custo(i)

# numero de geracoes
num_gera = 20

erro = list()

for g in tqdm(range(num_gera), leave=True, ascii=True, ncols=100):
    
    for e, i in enumerate(xi):
        # atualização das posições e velocidades das partículas
        # números pseudo-aleatórios uniformemente distribuídos
        r1 = np.random.rand()
        r2 = np.random.rand()
        vi[e] = w * vi[e] + c1 * r1 * (pi[e] - i) + c2 * r2 * (pg - i)
        i = i + vi[e]

        # calculo da funcao custo e atualização
        # dos menores valores globais e individuais
        p1 = func_custo(i)
        p2 = func_custo(pi[e])
        if p1 < p2:
            pi[e] = i
            p3 = func_custo(pg)
            if p1 < p3:
                pg = i


x = np.arange(0, 15, 0.01)

pol_1 = np.poly1d(pg)

# ph = np.polyfit(aero[:, 0], aero[:, 1], 4)
# pol_2 = np.poly1d(ph)

plt.plot(aero[:, 0], aero[:, 1], 'go', x, pol_1(x), 'r--', lw=3.0)
plt.xlabel('velocidade do vento (m/s)')
plt.ylabel('potencia gerada (kW)')
plt.grid(True)
plt.legend(['Conj. de dados aerogerador',
            'Polinomino gerado via DE'])
plt.show()
