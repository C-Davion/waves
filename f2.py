
from scipy import signal
from math import *
from data import *
import cmath
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar,minimize,Bounds



# on notera w pour omega
def g1(y,w,amplitude=1.0):  # gausienne pour modéliser un bruit localiser
    try:
        if -L <= y <= L:
            return amplitude * sin(w * y)
        else:
            raise ValueError("Error: y is outside the valid range [-L, L]")
    except ValueError as e:
        return str(e)


def g2(y,w, amplitude=1.0): # 1 si w in [20,20000], pour modéliser une salle de classe
    try:
        if -L<=y<=L:
            freq=w/(2*pi)
            if 20<=freq<=20000:
                return 1    
            else: return 0
        else: 
            raise ValueError("Error: y is outside the valid range [-L, L]")
    except ValueError as e:
        return str(e)

def g3(y,w, amplitude =1.0,w0=1000,sigma=100,k_sirène=0.1): #sirène , sous la forme d'un signal carr;e
     #la sirène se trouve entre -k_sirène*L et k_sirène*L
    if -k_sirène*L<=y<=k_sirène*L:
        return amplitude *exp(-(w-w0)**2/sigma)
    else:
        return 0
    

def fourrier_k(g,w,k): #calcul le coefficient k de la transformée de fourrier de g 
    def gprime(y):
        return g(y,w) * np.exp(-complex(0,1) * np.pi * k * y /L) #le calcul du coefficient de fourrier nous fait calculer l'intégrale de cette fonction entre -L et L
    result, error = quad(gprime, -L, L)
    return result




def calculate_e_k(alpha, w, k,g,A=1,B=1):
    lambda0=f_lambda0(k,w)
    lambda1=f_lambda1(k,w)
    gk=fourrier_k(g,w,k)
    chi=f_chi(k, lambda0, lambda1, gk,alpha)
    gamma=f_gamma(k, alpha, lambda0,lambda1, gk)
    
    if k**2 >= Ksi0 / eta0 * w**2:
        result = (A + B * k**2) * (L * (abs(chi)**2 + abs(gamma)**2) +
                 (1j / lambda0) * (chi.conjugate() * gamma * (1 - cmath.exp(-2 * lambda0 * L))))
        result += B * L * abs(lambda0)**2 * (abs(chi)**2 + abs(gamma)**2) + 1j * B * lambda0 * (chi.conjugate() * gamma * (1 - cmath.exp(-2 * lambda0 * L)))
    else:
        result = (A + B * k**2) * (1 / (2 * lambda0) * ((abs(chi)**2 * (1 - cmath.exp(-2 * lambda0 * L))) + (abs(gamma)**2 * (cmath.exp(2 * lambda0 * L) - 1))) + 2 * L * (chi.real * gamma.real))
        result += B * (lambda0 / 2) * ((abs(chi)**2 * (1 - cmath.exp(-2 * lambda0 * L)) + abs(gamma)**2 * (cmath.exp(2 * lambda0 * L) - 1)) - 2 * B * lambda0**2 * L * (chi.real * gamma.real))
    return result


def e(alpha,w,N,g):
    result=0
    for i in range(-N,N+1):
        k=i*np.pi/L
        result+=calculate_e_k(alpha,w,k,g)
    return result




def optimal_alpha(w,N,g):
    init=[w/10,w/10]
    def optimize_e(xy):
        x,y=xy
        return abs(e(x+1j*y,w,N,g))
    result = minimize(optimize_e,init,method='L-BFGS-B')
    return result


#print(optimal_alpha(1001,15,g3))








def plooting():
    w_range=np.linspace(600,30000,100)
    alpha_real=[]
    alpha_img=[]
    for w in w_range:
        alpha_opti=optimal_alpha(w,15,g3)
        alpha_real.append(alpha_opti.x[1])
    plt.plot(w_range,alpha_real,label='Im(a)')
    plt.legend()
    plt.show()


plooting()