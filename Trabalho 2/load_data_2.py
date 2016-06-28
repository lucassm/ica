"""Load Data."""

import numpy as np
import matplotlib.pyplot as plt

f = open('Gauss3.dat.txt')
x = list()
y = list()
for i, j in enumerate(f.readlines()):
    if i > 61:
        y.append(float(j[0:11]))
        x.append(float(j[12:21]))
x = np.array(x)
y = np.array(y)

plt.plot(x, y, 'ro')

# para uma aproximacao linear por partes do modelo 
# se faz necessario dividi-lo em cinco intervalos, a saber:
# 0   < x < 70  (Intervalo 1)
# 70  < x < 110 (Intervalo 2)
# 110 < x < 140 (Intervalo 3)
# 140 < x < 180 (Intervalo 4)
# 180 < x < 250 (Intervalo 5)

# definicao dos limites
a = 75.0
b = 110.0
c = 150.0
d = 175.0
e = 250.0

x1, y1 = list(), list()
x2, y2 = list(), list()
x3, y3 = list(), list()
x4, y4 = list(), list()
x5, y5 = list(), list()


for i, j in zip(x, y):
    if i <= a:
        x1.append(i)
        y1.append(j)
    elif i > a and i<= b:
        x2.append(i)
        y2.append(j)
    elif i > b and i <= c:
        x3.append(i)
        y3.append(j)
    elif i > c and i <= d:
        x4.append(i)
        y4.append(j)
    elif i > d and i <= e:
        x5.append(i)
        y5.append(j)

g = 1
p1 = np.polyfit(x1, y1, g)
pi1 = np.poly1d(p1)

p2 = np.polyfit(x2, y2, g)
pi2 = np.poly1d(p2)

p3 = np.polyfit(x3, y3, g)
pi3 = np.poly1d(p3)

p4 = np.polyfit(x4, y4, g)
pi4 = np.poly1d(p4)

p5 = np.polyfit(x5, y5, g)
pi5 = np.poly1d(p5)

plt.plot(x1, pi1(x1), 'b--', lw=3.0)
plt.plot(x2, pi2(x2), 'b--', lw=3.0)
plt.plot(x3, pi3(x3), 'b--', lw=3.0)
plt.plot(x4, pi4(x4), 'b--', lw=3.0)
plt.plot(x5, pi5(x5), 'b--', lw=3.0)

##############################
# Modelo de Inferencia Fuzzy #
##############################

# -------------------------------------
# Definicao das FUNCOES DE PERTINENCIAS
# -------------------------------------

def pert_triang(a, b, m, x):
    if x < a or x > b:
        return 0.0
    elif x >= a and x <= m:
        return (x - a) / (m - a)
    elif x >= m and x <= b:
        return (b - x) / (b - m)


def pert_trape_lat_dir(a, m, x):
    if x < a:
        return 0.0
    elif x >= a and x <= m:
        return (x - a) / (m - a)
    elif x >= m:
        return 1.0


def pert_trape_lat_esq(m, b, x):
    if x < m:
        return 1.0
    elif x >= m and x <= b:
        return (b - x) / (b - m)
    elif x >= b:
        return 0.0

def calcula_pertinencias(x):
    pertinencias = dict()
    # limites
    pertinencias['B'] = pert_trape_lat_esq(0.0, 110.0, x)
    pertinencias['M'] = pert_triang(40.0, 140.0, 100.0, x)
    pertinencias['A'] = pert_trape_lat_dir(120.0, 160.0, x)
    return pertinencias

y_ = np.arange(0.0, 250.0, 0.1)
x_ = list()

for i in y_:

    # Calculo dos valores de y
    y1 = pi1(i)
    y2 = pi2(i)
    y3 = pi4(i)

    # Defusificacao (Media ponderada Simples)
    pert = calcula_pertinencias(i)
    m = sum([pert['B'] * y1, pert['M'] * y2, pert['A'] * y3]) / sum(pert.values())
    x_.append(m)

# plt.plot(y_, x_, 'k--', lw=3.0)
plt.show()
