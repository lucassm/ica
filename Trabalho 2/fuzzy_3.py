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

plt.subplot(211)
plt.plot(x, y, 'ro')

# para uma aproximacao linear por partes do modelo 
# se faz necessario dividi-lo em cinco intervalos, a saber:
# 0   < x < 70  (Intervalo 1)
# 70  < x < 110 (Intervalo 2)
# 110 < x < 140 (Intervalo 3)
# 140 < x < 180 (Intervalo 4)
# 180 < x < 250 (Intervalo 5)

# definicao dos limites
a = 90.0
b = 180.0
c = 250.0

x1, y1 = list(), list()
x2, y2 = list(), list()
x3, y3 = list(), list()


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

g = 3
p1 = np.polyfit(x1, y1, g)
pi1 = np.poly1d(p1)

p2 = np.polyfit(x2, y2, g)
pi2 = np.poly1d(p2)

p3 = np.polyfit(x3, y3, g)
pi3 = np.poly1d(p3)


# plt.plot(x1, pi1(x1), 'g-', lw=3.0)
# plt.plot(x2, pi2(x2), 'g-', lw=3.0)
# plt.plot(x3, pi3(x3), 'g-', lw=3.0)

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


def pert_gauss(m, s, x):
    return 1/np.sqrt(2*np.pi*s**2) * np.exp(-(x-m)**2/(2*s**2))


def calcula_pertinencias_1(x):
    pertinencias = dict()
    # limites
    pertinencias['B'] = pert_trape_lat_esq(0.0, 120.0, x)
    pertinencias['M'] = pert_triang(90.0, 185.0, 120.0, x)
    pertinencias['A'] = pert_trape_lat_dir(170.0, 220.0, x)
    return pertinencias

def calcula_pertinencias_2(x):
    pertinencias = dict()
    # limites
    pertinencias['B'] = pert_gauss(50.0, 30.0, x)
    pertinencias['M'] = pert_gauss(140.0, 30.0, x)
    pertinencias['A'] = pert_gauss(220.0, 30.0, x)
    return pertinencias

y_ = np.arange(0.0, 250.0, 0.1)
x_ = list()

for i in y_:

    # Calculo dos valores de y
    y1 = pi1(i)
    y2 = pi2(i)
    y3 = pi3(i)

    # Defusificacao (Media ponderada Simples)
    pert = calcula_pertinencias_1(i)
    m = sum([pert['B'] * y1, pert['M'] * y2, pert['A'] * y3]) / sum(pert.values())
    x_.append(m)

plt.plot(y_, x_, 'k--', lw=3.0)

pert = dict()
pert['B'] = [calcula_pertinencias_1(i)['B'] for i in y_]
pert['M'] = [calcula_pertinencias_1(i)['M'] for i in y_]
pert['A'] = [calcula_pertinencias_1(i)['A'] for i in y_]

# pert = calcula_pertinencias_2(y_)

plt.subplot(212)
plt.plot(y_, pert['B'])
plt.plot(y_, pert['M'])
plt.plot(y_, pert['A'])

plt.show()
