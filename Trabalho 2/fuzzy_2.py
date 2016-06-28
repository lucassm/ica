"""
Questao 1 do Trabalho 02 de ICA.

Exemplo de Estacionamento de veiculo.
"""

import numpy as np
import matplotlib.pyplot as plt

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


def distancia(x):
    pertinencias = dict()
    # conjunto de distancias CE
    pertinencias['CE'] = pert_triang(45.0, 55.0, 50.0, x)
    # conjunto de distancias RC
    pertinencias['RC'] = pert_triang(50.0, 70.0, 60.0, x)
    # conjunto de distancias LC
    pertinencias['LC'] = pert_triang(30.0, 50.0, 40.0, x)
    # conjunto de distancias LE
    pertinencias['LE'] = pert_trape_lat_esq(10.0, 35.0, x)
    # conjunto de distancias RI
    pertinencias['RI'] = pert_trape_lat_dir(65.0, 90.0, x)
    return pertinencias


def angulo_phi(phi):
    pertinencias = dict()
    # conjunto de angulos VE
    pertinencias['VE'] = pert_triang(80.0, 100.0, 90.0, phi)
    # conjunto de angulos RV
    pertinencias['RV'] = pert_triang(45.0, 90.0, 62.5, phi)
    # conjunto de angulos RU
    pertinencias['RU'] = pert_triang(0.0, 60.0, 30.0, phi)
    # conjunto de angulos RB
    pertinencias['RB'] = pert_triang(-100.0, 10.0, -50.0, phi)
    # conjunto de angulos LV
    pertinencias['LV'] = pert_triang(90.0, 135.0, 112.5, phi)
    # conjunto de angulos LU
    pertinencias['LU'] = pert_triang(120.0, 180.0, 150.0, phi)
    # conjunto de angulos LB
    pertinencias['LB'] = pert_triang(160.0, 280.0, 220.0, phi)
    return pertinencias


def angulo_theta(theta):
    pertinencias = dict()
    # conjunto de angulos ZE
    pertinencias['ZE'] = pert_triang(-5.0, 5.0, 0.0, theta)
    # conjunto de angulos PS
    pertinencias['PS'] = pert_triang(0.0, 10.0, 5.0, theta)
    # conjunto de angulos PM
    pertinencias['PM'] = pert_triang(5.0, 25.0, 15.0, theta)
    # conjunto de angulos PB
    pertinencias['PB'] = pert_trape_lat_dir(15.0, 30.0, theta)
    # conjunto de angulos NS
    pertinencias['NS'] = pert_triang(-10.0, 0.0, -5.0, theta)
    # conjunto de angulos NM
    pertinencias['NM'] = pert_triang(-25.0, -5.0, -15.0, theta)
    # conjunto de angulos NB
    pertinencias['NB'] = pert_trape_lat_esq(-30.0, -15.0, theta)
    return pertinencias


def regras(mi1, mi2, mi_out, y):
    RULE_OUT = list()
    RULE_OUT_M = list()

    # regra 1: se LE e RB entao PS
    m1 = min(mi1['LE'], mi2['RB'])
    mi_out_R1 = [min(m1, i['PS']) for i in mi_out]
    RULE_OUT.append(mi_out_R1)
    RULE_OUT_M.append(m1)
    # regra 2: se LC e RB entao PM
    m2 = min(mi1['LC'], mi2['RB'])
    mi_out_R2 = [min(m2, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R2)
    RULE_OUT_M.append(m2)

    # regra 3: se CE e RB entao PM
    m3 = min(mi1['CE'], mi2['RB'])
    mi_out_R3 = [min(m3, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R3)
    RULE_OUT_M.append(m3)
    
    # regra 4: se RC e RB entao PB
    m4 = min(mi1['RC'], mi2['RB'])
    mi_out_R4 = [min(m4, i['PB']) for i in mi_out]
    RULE_OUT.append(mi_out_R4)
    RULE_OUT_M.append(m4)

    # regra 5: se RI e RB entao PB
    m5 = min(mi1['RI'], mi2['RB'])
    mi_out_R5 = [min(m5, i['PB']) for i in mi_out]
    RULE_OUT.append(mi_out_R5)
    RULE_OUT_M.append(m5)

    # regra 6: se LE e RU entao NS
    m6 = min(mi1['LE'], mi2['RU'])
    mi_out_R6 = [min(m6, i['NS']) for i in mi_out]
    RULE_OUT.append(mi_out_R6) 
    RULE_OUT_M.append(m6)

    # regra 7: se LC e RU entao PS
    m7 = min(mi1['LC'], mi2['RU'])
    mi_out_R7 = [min(m7, i['PS']) for i in mi_out]
    RULE_OUT.append(mi_out_R7)
    RULE_OUT_M.append(m7)
    
    # regra 8: se CE e RU entao PM
    m8 = min(mi1['CE'], mi2['RU'])
    mi_out_R8 = [min(m8, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R8)
    RULE_OUT_M.append(m8)
    
    # regra 9: se RC e RU entao PB
    m9 = min(mi1['RC'], mi2['RU'])
    mi_out_R9 = [min(m9, i['PB']) for i in mi_out]
    RULE_OUT.append(mi_out_R9)
    RULE_OUT_M.append(m9)
    
    # regra 10:se RI e RU entao PB
    m10 = min(mi1['RI'], mi2['RU'])
    mi_out_R10 = [min(m10, i['PB']) for i in mi_out]
    RULE_OUT.append(mi_out_R10)
    RULE_OUT_M.append(m10)


    # regra 11: se LE e RV entao NM
    m11 = min(mi1['LE'], mi2['RV'])
    mi_out_R11 = [min(m11, i['NM']) for i in mi_out]
    RULE_OUT.append(mi_out_R11)
    RULE_OUT_M.append(m11)

    # regra 12: se LC e RV entao NS
    m12 = min(mi1['LC'], mi2['RV'])
    mi_out_R12 = [min(m12, i['NS']) for i in mi_out]
    RULE_OUT.append(mi_out_R12)
    RULE_OUT_M.append(m12)

    # regra 13: se CE e RV entao PS
    m13 = min(mi1['CE'], mi2['RV'])
    mi_out_R13 = [min(m13, i['PS']) for i in mi_out]
    RULE_OUT.append(mi_out_R13)
    RULE_OUT_M.append(m13)

    # regra 14: se RC e RV entao PM
    m14 = min(mi1['RC'], mi2['RV'])
    mi_out_R14 = [min(m14, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R14)
    RULE_OUT_M.append(m14)

    # regra 15: se RI e RV entao PB
    m15 = min(mi1['RI'], mi2['RV'])
    mi_out_R15 = [min(m15, i['PB']) for i in mi_out]
    RULE_OUT.append(mi_out_R15)
    RULE_OUT_M.append(m15)


    # regra 16: se LE e VE entao NM
    m16 = min(mi1['LE'], mi2['VE'])
    mi_out_R16 = [min(m16, i['NM']) for i in mi_out]
    RULE_OUT.append(mi_out_R16)
    RULE_OUT_M.append(m16)

    # regra 17: se LC e VE entao NM
    m17 = min(mi1['LE'], mi2['RB'])
    mi_out_R17 = [min(m17, i['PS']) for i in mi_out]
    RULE_OUT.append(mi_out_R17)
    RULE_OUT_M.append(m17)

    # regra 18: se CE e VE entao ZE
    m18 = min(mi1['CE'], mi2['VE'])
    mi_out_R18 = [min(m18, i['ZE']) for i in mi_out]
    RULE_OUT.append(mi_out_R18)
    RULE_OUT_M.append(m18)

    # regra 19: se RC e VE entao PM
    m19 = min(mi1['RC'], mi2['VE'])
    mi_out_R19 = [min(m19, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R19)
    RULE_OUT_M.append(m19)

    # regra 20: se RI e VE entao PM
    m20 = min(mi1['RI'], mi2['VE'])
    mi_out_R20 = [min(m20, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R20)
    RULE_OUT_M.append(m20)


    # regra 21: se LE e LV entao NB
    m21 = min(mi1['LE'], mi2['LV'])
    mi_out_R21 = [min(m21, i['NB']) for i in mi_out]
    RULE_OUT.append(mi_out_R21)
    RULE_OUT_M.append(m21)

    # regra 22: se LC e LV entao NM
    m22 = min(mi1['LC'], mi2['LV'])
    mi_out_R22 = [min(m22, i['NM']) for i in mi_out]
    RULE_OUT.append(mi_out_R22)
    RULE_OUT_M.append(m22)

    # regra 23: se CE e LV entao NS
    m23 = min(mi1['CE'], mi2['LV'])
    mi_out_R23 = [min(m23, i['NS']) for i in mi_out]
    RULE_OUT.append(mi_out_R23)
    RULE_OUT_M.append(m23)

    # regra 24: se RC e LV entao PS
    m24 = min(mi1['RC'], mi2['LV'])
    mi_out_R24 = [min(m24, i['PS']) for i in mi_out]
    RULE_OUT.append(mi_out_R24)
    RULE_OUT_M.append(m24)

    # regra 25: se RI e LV entao PM
    m25 = min(mi1['RI'], mi2['LV'])
    mi_out_R25 = [min(m25, i['PM']) for i in mi_out]
    RULE_OUT.append(mi_out_R25)
    RULE_OUT_M.append(m25)


    # regra 26: se LE e LU entao NB
    m26 = min(mi1['LE'], mi2['LU'])
    mi_out_R26 = [min(m26, i['NB']) for i in mi_out]
    RULE_OUT.append(mi_out_R26)
    RULE_OUT_M.append(m26)

    # regra 27: se LC e LU entao NB
    m27 = min(mi1['LC'], mi2['LU'])
    mi_out_R27 = [min(m27, i['NB']) for i in mi_out]
    RULE_OUT.append(mi_out_R27)
    RULE_OUT_M.append(m27)

    # regra 28: se CE e LU entao NM
    m28 = min(mi1['CE'], mi2['LU'])
    mi_out_R28 = [min(m28, i['NM']) for i in mi_out]
    RULE_OUT.append(mi_out_R28)
    RULE_OUT_M.append(m28)

    # regra 29: se RC e LU entao NS
    m29 = min(mi1['RC'], mi2['LU'])
    mi_out_R29 = [min(m29, i['NS']) for i in mi_out]
    RULE_OUT.append(mi_out_R29)
    RULE_OUT_M.append(m29)

    # regra 30: se RI e LU entao PS
    m30 = min(mi1['RI'], mi2['LU'])
    mi_out_R30 = [min(m30, i['PS']) for i in mi_out]
    RULE_OUT.append(mi_out_R30)
    RULE_OUT_M.append(m30)


    # regra 31: se LE e LB entao NB
    m31 = min(mi1['LE'], mi2['LB'])
    mi_out_R31 = [min(m31, i['NB']) for i in mi_out]
    RULE_OUT.append(mi_out_R31)
    RULE_OUT_M.append(m31)

    # regra 32: se LC e LB entao NB
    m32 = min(mi1['LC'], mi2['LB'])
    mi_out_R32 = [min(m32, i['NB']) for i in mi_out]
    RULE_OUT.append(mi_out_R32)
    RULE_OUT_M.append(m32)

    # regra 33: se CE e LB entao NM
    m33 = min(mi1['CE'], mi2['LB'])
    mi_out_R33 = [min(m33, i['NM']) for i in mi_out]
    RULE_OUT.append(mi_out_R33)
    RULE_OUT_M.append(m33)

    # regra 34: se RC e LB entao NM
    m34 = min(mi1['RC'], mi2['LB'])
    mi_out_R34 = [min(m34, i['NM']) for i in mi_out]
    RULE_OUT.append(mi_out_R34)
    RULE_OUT_M.append(m34)

    # regra 35: se RI e LB entao NS
    m35 = min(mi1['RI'], mi2['LB'])
    mi_out_R35 = [min(m35, i['NS']) for i in mi_out]
    RULE_OUT.append(mi_out_R35)
    RULE_OUT_M.append(m35)


    return RULE_OUT, RULE_OUT_M


def calc_CG(mi_out, y):
    if sum(mi_out) != 0.0:
        return sum(mi_out * y) / sum(mi_out)
    else:
        return 0.0


# valores medidos das variaveis de entrada
x = 47.5
phi = 99.0

#############################
# # Etapa 1: FUZZIFICACAO # #
#############################
mi1 = distancia(x)  # pertinencias para variavel DISTANCIA
mi2 = angulo_phi(phi)  # pertinencias para variavel ANGULO

# VARIAVEL DE SAIDA: funcoes de pertinencia
y = np.arange(-30.0, 30.0, 0.1)  # Universo de discurso da variavel de saida
mi_out = [angulo_theta(i) for i in y]

###########################################
# # Etapa 2: AVALIACAO DAS REGRAS FUZZY # #
###########################################

# Conjuntos fuzzy de saida de todas as regras
rules_out, rules_out_m = regras(mi1, mi2, mi_out, y)
rules_out,rules_out_m = np.array(rules_out), np.array(rules_out_m)


# Desenha o grafico das regras que foram ativadas
# pelos valores das variaveis de entrada
rules_cg = list()
plt.subplot(211)
for rule in rules_out:
    rules_cg.append(calc_CG(rule, y))
    if max(rule) > 0.0:
        plt.plot(y, rule)


################################
# # Etapa 4: DESFUZZIFICACAO # #
################################
rules_out_m = np.array(rules_out_m)
rules_cg = np.array(rules_cg)

Y=sum(rules_out_m*rules_cg)/sum(rules_out_m)

print Y
