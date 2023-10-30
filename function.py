import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
import scipy
from source import g1,g2,g3

#on python 3.12
A=1
B=1
L=10
poro=0.99
resitance=14000
tort=1.02
gp=7/5
c0=340 #vitesse du son dans l'air en m/s
p0=1.292 # masse vol de l'air en kg/m^3
deltax=0.01


e0=1/(c0**2)
n0=1
a0=0
e1=poro*gp/(c0**2)
a1=(resitance*poro**2*gp)/(c0**2*p0*tort)
n1=poro/tort


def lambda0(k,w):
    k2=k**2
    w2=e0/n0*(w**2)
    if k2>=w2:
        return complex(np.sqrt(k2-w2),0)
    else:
        return complex(0,np.sqrt(w2-k2))
    


def lambda1(k,w):
    k2=k**2 #holder
    w2=e1/n1*w**2
    diff=k2-w2
    var1=(a1/n1*w)**2 #holder
    var2=sqrt(diff**2+var1)
    real=sqrt(diff+var2)
    real=1/sqrt(2)*real
    im=sqrt(-diff+var2)
    im=-1/sqrt(2)*im
    return complex(real,im)

def f(x,k,w):
    l0=lambda0(k,w)
    l0n0=l0*n0
    l0L=l0*L
    temp1=(l0n0-x)*np.exp(-l0L)
    temp2=(l0n0+x)*np.exp(l0L)
    return temp1+temp2


def chi_and_gamma(k,alpha,w,gk):
    l0=lambda0(k,w) #holder, could be optimize 
    l1=lambda1(k,w)
    l0n0=l0*n0
    l1n1=l1*n1
    f1=f(alpha,k,w)
    f2=f(l1n1,k,w)
    chi=(l0n0-l1n1)/f2-(l0n0-alpha)/f1
    chi=gk*chi
    gamma= (l0n0+l1n1)/f2-(l0n0+alpha)/f1
    gamma=gk*gamma
    return chi,gamma

def fourier(g,k,w):
    var=complex(0,1)
    error=complex(0,0)
    def temp(y):
        return g(y,w)*np.exp(-var*np.pi*k*y/L) #calcul du k-ieme coeff de fourier
    res,error=scipy.integrate.quad(temp,-L,L)
    return complex(res,0)


def compute_fourier_coefficient(g,k, w):
    # Define the integrand for the Fourier coefficient
    integrand = lambda y: g(y, w) * np.exp(-1j * k * (2 * np.pi / (2 * L) * y))

    # Perform the integration over the range [-L, L]
    result, _ = scipy.integrate.quad(integrand, -L, L)

    # Divide by 2*L to compute the coefficient
    ck = complex(result.real / (2 * L), result.imag / (2 * L))

    return ck

def ek(alpha,k,w,gk):
    l0=lambda0(k,w)
    chi,gamma=chi_and_gamma(k,alpha,w,gk)
    coeff=(A+B*k**2)
    var1=(1+np.exp(-2*l0*L))
    var2=(-1+np.exp(2*l0*L))
    if k**2>=e0/n0*w**2:
        temp1=chi*chi.conjugate()*var1+gamma*gamma.conjugate()*var2
        temp2=temp1
        prod=chi*gamma.conjugate()
        temp1=1/(2*l0)*temp1+2*L*prod.real
        temp1=coeff*temp1
        temp2=B*l0/2*temp2-2*B*l0**2*L*prod.real
        return temp1+temp2
    else:
        
        r1=L*(chi*chi.conjugate()+gamma*gamma.conjugate())
        i1=chi*gamma.conjugate()*var1
        i1=i1.imag
        i3=B*l0*i1
        i1=1/l0*i1
        temp1=complex(r1,i1)
        temp1=coeff*temp1
        temp2=B*l0*l0.conjugate()*r1
        temp3=complex(0,i3)
        return temp1+temp2+temp3


def sum_ek(alpha,w,g):
    res=complex(0,0)
    floored=int(L/deltax)
    for n in range(-floored,floored+1): # summing over too much points leads to floating point overflow. With 5, it leads to a lesser precision.
        k=n*np.pi/L
        gk=compute_fourier_coefficient(g,k,w)
        res+=ek(alpha,k,w,gk)
    return res


def minimise(): #modify the function with the correct source, do not forget to modify the labels as well
    initial_alpha=[0,0]
    w_range=np.linspace(600,30000,100)
    g=g1
    alpha_real=[]
    alpha_img=[]
    for w in w_range:
        def objective_function(xy):
            x,y=xy
            res=sum_ek(x+1j*y,w,g2)
            return np.abs(res)
        result=scipy.optimize.minimize(objective_function,initial_alpha,method='L-BFGS-B')
        alpha_real.append(result.x[0])
        alpha_img.append(result.x[1])
    
    plt.plot(w_range, alpha_real, color='red', label='partie Reele de alpha pour g2')  
    plt.xlabel('w')
    plt.ylabel('Re(a)')
    plt.legend()

    plt.show()
''''
def optimal_alpha(w,N,g):
    init=[0,0]
    def optimize_e(xy):
        x,y=xy
        return abs(e(x+1j*y,w,N,g))
    result = minimize(optimize_e,init,method='L-BFGS-B')
    return result
'''



minimise()
#print(compute_fourier_coefficient(2,L,1000))



    
