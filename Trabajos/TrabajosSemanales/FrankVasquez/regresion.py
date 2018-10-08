# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:46:29 2018

@author: Frank Vasquez
"""

import pandas as pd
import math as mt
import os
import numpy as np
import matplotlib.pyplot as plt

x = np.array([14,7,13,12,16,14,18,13,12,16,13]) # Valor de x
y = np.array([16,12,13,14,15,12,16,11,13,18,17]) # Valor de y

covXY = 0 # Covarianza 
PromX = 0# Promedio de X
promY = 0# Promdio de Y
desvX = 0# Desviacion de X
desvY = 0# Desviacion de Y
sx = 0 # Sumatoria de X
sy = 0 # Sumatoria de Y
sxy = 0 # Sumatoria de los XY
sx2 = 0 # Sumatoria de los x al cuadrado
sy2 =0# Sumatoria de los y al cuadrado
r =0 # Correlacion de pearson
longitud  = 11
xy = [] # Valor de los xy
x2 = [] # Valor de los x al cuadrado
y2 = [] # Valor de los y al cuadrado
m =0 # Intercepto
b =0 # Corte de la recta

#Lectura de los datos


#Calcular Matriz
for i in range(longitud):
    xy.append(x[i]*y[i])
    x2.append(x[i]**2)
    y2.append(y[i]**2)
    i+=1
for i in range(longitud):
    sx = sx + x[i]
    sy = sy + y[i]
    sxy = sxy + xy[i]
    sx2 = sx2 + x2[i]
    sy2 = sy2 + y2[i]   
    i+=1
#Calcular Promedio
PromX = sx/longitud 
PromY = sy/longitud

#Calcular Covarianza
covXY = (sxy/longitud) - (PromX*PromY)

    
#Calcular Desviaciones
desvX = mt.sqrt((sx2/longitud) - (PromX**2))
desvY = mt.sqrt((sy2/longitud) - (PromY**2))

# CalcularR():
r = covXY/(desvX*desvY)

# Pendiente():
m = (sxy-(sx*sy/longitud))/(sx2-((sx2**2)/longitud))


# Corte_Recta():
b = PromY - (m*PromX)

NotaM = int(input("Ingrese Nota de Matematica: "))
PredNP = (covXY/desvX**2)*(NotaM-PromX)+PromY
print (f'{"La nota de Programacion esperada es: "} {PredNP}')

NotaP = int(input("Ingrese Nota de Programacion: "))
PredNM = (covXY/desvY**2)*(NotaP-PromY)+PromX
print (f'{"La nota de Matematica esperada es: "} {PredNM}')

print("Nota de Programacion Aplicando minimos cuadrados")
PredNP2 = m*NotaM + b

print (PredNP2)
#print("El promedio de X es {}".format(PromX))
#print("El promedio de Y es {}".format(PromY))
#print("La covarianza es es {}".format(covXY))
#print("La desviacion de X es {}".format(desvX))
#print("La desviacion de Y es {}".format(desvY))
#print("El coeficiente de correlacion es {}".format(r))
#print("La pendiente es {}".format(m))
#print("El corte de la recta  es {}".format(b))

print("Diagrama de dispersion")

plt.scatter(x,y)