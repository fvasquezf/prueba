import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import math


print("Coordenada A")
A = (int(input("Ingrese X1: ")),int(input("Ingrese Y1: ")))
print("Coordenada B")
B = (int(input("Ingrese X2: ")),int(input("Ingrese Y2: ")))
print("Coordenada B")
C = (int(input("Ingrese X3: ")),int(input("Ingrese Y3: ")))

plt.axis([0, 10, 0, 10])

plt.plot((A[0], B[0], B[0]), (A[1], A[1], B[1]), (B[0], C[0], C[0]), (B[1], B[1], C[1]), '.-', markevery=(0, 2))
plt.show()

distancia_cos = dot(A, B)/(norm(A)*norm(B))
distancia_man = abs(abs(B[0]-A[0])+abs(B[1]-A[1])+abs(C[0]-B[0])+abs(C[1]-B[1]))
distancia_euc = math.sqrt((B[0]-A[0])^2)+((B[1]-A[1])^2)
print(f'{"La distancia Manhattan es: "}{distancia_man}')
print(f'{"La distancia Coseno es: "}{distancia_cos}')
print(f'{"La distancia Euclidiana es: "}{distancia_euc}')
