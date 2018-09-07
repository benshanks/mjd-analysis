import matplotlib.pyplot as plt
import numpy as np

def main():
    t = np.linspace(00,5.1, 10000)

    zero = -0.3
    pole = -0.31
    lp_rc = -100

    for i in (0.5, 0.75, 1, 1.25, 1.5):
        l = lp_rc*i
        os = get_overshoot(t,zero,pole, l)
        plt.plot(t/10*1E3,os, label="{}".format(l))
    plt.legend()
    plt.show()

def get_overshoot(t, a,b,r):
    return (-b*-r*-r/-a)*(np.exp(r*t)*(-a*b +2*a*r-r**2)/(r**2*(b-r)**2)
        +   a/(b*r**2)
        +(t * (a-r)*np.exp(r*t))/(r*(b-r))
        +(b-a)*np.exp(b*t)/(b*(b-r)**2))

if __name__=="__main__":
    main()
