from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
#from sympy.physics.quantum import TensorProduct, Dagger
import numpy as np
from numpy import linspace
#import matplotlib.pyplot as plt
#import math
#from theoric.tools import *
#import torch
from torch import tensor

class QuantumChannels(object):
    def __init__(self):
        theta = Symbol('theta',real=True)
        phi = Symbol('phi',real=True)
        gamma = Symbol('gamma',real=True, positive=True)
        p = Symbol('p',real=True, positive=True)

    def get_target_op(state):
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        state2 = np.zeros(np.shape(rho)[1],dtype=complex)
        aux = 0
        for i in rho[0]:
            state2[aux] = i
            aux += 1
        target_op = np.outer(state2.conj(), state2)
        target_op = tensor(target_op)
        return target_op
    
    def rho_AB_ad(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        0]])
        return state
    
    def rho_AB_bpf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*(cos(theta/2))),
                        (-1j*sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        (1j*sqrt(p)*cos(theta/2))]])
        return state

    def rho_AB_bf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*cos(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        sqrt(p)*cos(theta/2)]])
        return state

    def rho_AB_pf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*cos(theta/2)),
                        -(sqrt(p)*1j*sin(theta/2)),
                        (sqrt(p)*1j*cos(theta/2) +sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        0]])
        return state

    def rho_AB_pd(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                         0,
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2))]])
        return state
    
    @staticmethod
    def rho_AB_l(theta, phi, p):
        state = Matrix([[sqrt(1/2)*(cos(p/2)*cos(theta/2) + sin(p/2)*exp(1j*phi)*sin(theta/2)),
                         sqrt(1/2)*(cos(p/2)*cos(theta/2) - sin(p/2)*exp(1j*phi)*sin(theta/2)),
                         sqrt(1/2)*(cos(p/2)*exp(1j*phi)*sin(theta/2) - sin(p/2)*cos(theta/2)),
                         sqrt(1/2)*(cos(p/2)*exp(1j*phi)*sin(theta/2) + sin(p/2)*cos(theta/2))
                         ]])
        return state

    #def rho_AB_d_old(theta, phi, p):
    #    state = Matrix([[(sqrt(1-3*p/4)*cos(theta/2)),
    #                    (sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
    #                    -1j*(sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
    #                    (sqrt(p/4)*cos(theta/2)),
    #                    (sqrt(1-3*p/4)*exp(1j*phi)*sin(theta/2)),
    #                    (sqrt(p/4)*cos(theta/2)),
    #                    (1j*sqrt(p/4)*cos(theta/2)),
    #                    0]])
    #    return state
    def rho_AB_d(theta, phi, p):
        state = Matrix([[(sqrt(1-3*p/4)*cos(theta/2)),
                        (sqrt(1-3*p/4)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p/4)*cos(theta/2)),
                        (sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
                        (1j*sqrt(p/4)*cos(theta/2)),
                        (-1j*sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p/4)*cos(theta/2)),
                        -(sqrt(p/4)*exp(1j*phi)*sin(theta/2))
                        ]])
        return state
    @staticmethod
    def rho_AB_adg(theta, phi, p):#, gamma):
        gamma = 0.5
        state = Matrix([[sqrt(p)*cos(theta/2),
                         sqrt(p*gamma)*exp(1j*phi)*sin(theta/2), #|001\rangle),
                         sqrt((1-p)*(1-gamma))*cos(theta/2), #|010\rangle
                         0, # |011\rangle
                         sqrt(p*(1-gamma))*exp(1j*phi)*sin(theta/2), #|100\rangle),
                         0, #|101\rangle)
                         sqrt(1-p)*exp(1j*phi)*sin(theta/2), #|110\rangle)
                         sqrt((1-p)*gamma)*cos(theta/2) #|111\rangle)
                        ]])
        return state

    def show_eq(rho, theta=None, phi=None, gamma=None, p=None):
        if theta == None or phi == None or gamma == None or p == None:
            theta = Symbol('theta',real=True)
            phi = Symbol('phi',real=True)
            gamma = Symbol('gamma',real=True, positive=True)
            p = Symbol('p',real=True, positive=True)
        print(rho(theta, phi, p))

    def show_eq_with_gamma(rho, theta=None, phi=None, gamma=None, p=None):
        if theta == None or phi == None or gamma == None or p == None:
            theta = Symbol('theta',real=True)
            phi = Symbol('phi',real=True)
            gamma = Symbol('gamma',real=True, positive=True)
            p = Symbol('p',real=True, positive=True)
        print(rho(theta, phi, p, gamma))


def main():
    from sympy import Symbol,simplify,print_latex
    import sys
    sys.path.append('runtime-qiskit')
    sys.path.append('src')
    from theoric_channels import TheoricMaps
    a = TheoricMaps()
    theta = Symbol('theta',real=True)
    phi = Symbol('phi',real=True)
    gamma = Symbol('gamma',real=True, positive=True)
    p = Symbol('p',real=True, positive=True)
    from kraus_maps import QuantumChannels
    a = QuantumChannels()
    print(simplify(a.rho_AB_adg(theta,phi,p, gamma)))

if __name__ == "__main__":
    main()
#a=10
#QCH = QuantumChannels()
#a = QCH.rho_AB_bpf
#print(a(0,0,0))
#print(QCH.get_target_op(QCH.rho_AB_pd(pi/2,0,0)))
