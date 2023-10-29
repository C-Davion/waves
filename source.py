from numpy import pi,sin, exp
from scipy import signal

#g-> g(w,y)

#ek-> ek(w,y)
L=10

#q2

#urban noise/underwater noise/

# on notera w pour omega
def g1(y,w,amplitude=1.0,c=1.0):  # modéliser le bruit causer par une aile d'avion, la freq d'oscillation va dépendre du matériel de l'aile et de où on se situe 
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

def g3(y,w, amplitude =1.0,w0=100): #sirène , sous la forme d'un gaussienne fenetrée
    try:
        if -L<=y<=L :
            if -L/2 <=y <=L/2 :
                return amplitude* exp(-(w-w0)**2/2)
            else: return 0
        else:
            raise ValueError("Error: y is outside the valide range [-L,L]")
    except ValueError as e:
        return str(e)
    

