from numpy import pi,sin, exp,sqrt
from scipy import signal


L=10


# on notera w pour omega
def g1(y,w,amplitude=1.0,c=1):  # modéliser le bruit causer par une aile d'avion, la freq d'oscillation va dépendre du matériel de l'aile et de où on se situe 
    try:
        if -L <= y <= L:
            return amplitude * sin(w * y/c)
        else:
            raise ValueError("Error: y is outside the valid range [-L, L]")
    except ValueError as e:
        return str(e)


def g2(y,w, amplitude=1.0): # 1 si freq dans [20,20000], pour modéliser une salle de classe
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

def g3(y,w, amplitude =1.0,w0=3000,sigma=10): #sirène , sous la forme d'un gaussienne fenetrée
    try:
        if -L<=y<=L :
            if -L/2 <=y <=L/2 :
                amplitude=amplitude/(sqrt(2*pi)*sigma)
                return amplitude*exp(-0.5*((w-w0)/sigma)**2)
            else: return 0
        else:
            raise ValueError("Error: y is outside the valide range [-L,L]")
    except ValueError as e:
        return str(e)
    

