#from src.simulation_with_save import *
from numpy import cos, sin, exp, linspace, meshgrid, array, zeros, shape
def non_markov_t(lamb,t):
    result = 1 - exp(-lamb*t)*(cos(t/2)+lamb*sin(t/2))**2
    return result
def get_t_noMarkov(lamb, t):
    #lamb = 5
    #gamma_0 = 2.8
    t_noMarkov = []
    for p in t:
        t_noMarkov.append(non_markov_t(lamb,p))
    return t_noMarkov
size = 10

t = linspace(0,40,size)
lamb_list = linspace(0,0.5,size)
z = zeros((size,size))
#z = non_markov_t(lamb_list, t)
j = 0
k = 0
for p in t:
    for lamb in lamb_list:
        z[j,k] = non_markov_t(lamb,p)
        j+=1
    k+=1
    j=0
#        z.append(non_markov_t(lamb,p))
#z = array(z)
t, lamb_list = meshgrid(t,lamb_list)

print(shape(t))
print(shape(lamb_list))
print(shape(z))

import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(t, lamb_list,z)
ax.set_xlabel('t')
ax.set_ylabel('lambda')
ax.set_zlabel('p(t,lambda)')
plt.show()
