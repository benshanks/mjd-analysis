import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from siggen.electronics import DigitalFilter
from scipy import signal, optimize

def main():

    df = pd.read_csv("front_end_1k_to_1g.txt", sep="\s+")

    f = df.frequency
    f_min_idx = np.argmax(df.frequency > 10**6)
    # f_max_idx = -1
    f_max_idx = np.argmax(df.frequency > 5E8)

    freqs = df.frequency.values[f_min_idx:f_max_idx]
    volts = df.vm[f_min_idx:f_max_idx]/df.vm.max()

    mod1 = DigitalFilter(2)
    mod2 = DigitalFilter(2)
    mod1.num = [1,1]
    mod2.num = [1]

    def get_filter(mag, phi):
        ws = freqs *(np.pi /  0.5E9)

        mod1.set_poles(mag, phi)
        mod2.set_poles(mag, phi)

        h1 = get_freqz(mod1, w = ws )
        h2 = get_freqz(mod2, w = ws)

        return np.abs(h1)*np.abs(h2)

    def min_func(x):
        v = get_filter(*x)

        return np.sum((volts-v)**2)

    res = optimize.minimize(min_func, [0.9, 0.1*np.pi], method="L-BFGS-B", bounds=[(0,1), (0,np.pi)])

    v = get_filter(*res["x"])
    print(res["x"])

    # plt.figure()
    f,ax = plt.subplots(2,1, sharex=True)
    ax[0].semilogx(freqs, db(volts), label="EAGLE")
    ax[0].semilogx(freqs, db( v ), label="4th order best fit")
    ax[1].plot(freqs, db(volts)-db( v ))

    plt.legend()
    plt.show()

def db(voltage):
    return 10*np.log10(voltage)

def freq_to_rad(freq):
    nyq_freq = 0.5*1E9

    return freq*1E6 * (np.pi /nyq_freq)

def get_freqz(df, w):
    nyq_freq = 0.5*1E9

    num, den = df.num, df.den
    if np.sum(num) != 0:
        num_calc = num/(np.sum(num)/np.sum(den))
    else:
        num_calc=num

    _, h = signal.freqz(num_calc, den, worN=w )
    w_out = w/(np.pi /nyq_freq)
    return h

if __name__ == "__main__":
    main()
