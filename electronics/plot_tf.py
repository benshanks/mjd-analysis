import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from siggen.electronics import DigitalFilter
from scipy import signal, optimize
from siggen import PPC

def main():
    # fit_digitizer()
    # fit_first_stage()
    # fit_stage2_decay()
    # fit_second_stage_lowpass()
    plot_bode()
    # plot_effect()

    plt.show()

def plot_effect():
    det = PPC( os.path.join(os.environ["DATADIR"],  "siggen", "config_files", "bege.config"), wf_padding=1000)
    imp_avg = -2
    imp_grad = 1.2
    det.siggenInst.SetImpurityAvg(imp_avg, imp_grad)
    wf_compare=None
    f,ax = plt.subplots(2,1)
    for func in [fit_first_stage, fit_digitizer, fit_second_stage_lowpass]:
        func(det)
        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 250))
        p = ax[0].plot(wf_proc)
        if wf_compare is None:
            wf_compare = wf_proc
        else:
            ax[1].plot(wf_proc-wf_compare, c=p[0].get_color())



def plot_bode():
    f_1, fs = read_first_stage()
    f_2, n2, p2 = read_second_stage()
    f_d, d1, d2 = read_digitzer()

    plt.figure()
    plt.loglog(f_1, fs, label="1st stage")
    plt.loglog(f_2, p2, label="2nd stage")
    plt.loglog(f_d, d1, label="Digitizer")

    plt.loglog(f_2, fs*d1*p2, label="Total")
    # plt.loglog(f_d, d2)
    # plt.loglog(f_1, fs*d1)
    plt.axvline(0.5E8, ls=":", c="k")

    plt.legend()



def fit_digitizer(det=None):
    f, d1, d2 = read_digitzer()
    f_min_idx = np.argmax(f > 10**6)
    f_max_idx = np.argmax(f> 5E8)
    freqs = f.values[f_min_idx:f_max_idx]
    spice_volts = d1[f_min_idx:f_max_idx]

    mod1 = DigitalFilter(2)
    mod2 = DigitalFilter(2)
    mod1.num = [1,1]
    mod2.num = [1]

    if det is not None:
        det.AddDigitalFilter(mod1)
        det.AddDigitalFilter(mod2)

    def get_filter(mag, phi):
        ws = freqs *(np.pi /  0.5E9)

        mod1.set_poles(mag, phi)
        mod2.set_poles(mag, phi)

        h1 = get_freqz(mod1, w = ws )
        h2 = get_freqz(mod2, w = ws)

        return np.abs(h1)*np.abs(h2)

    res = optimize.minimize(lsq, [0.9, 0.1*np.pi], method="L-BFGS-B", args=(get_filter, spice_volts), bounds=[(0,1), (0,np.pi)])

    v = get_filter(*res["x"])
    print(res["x"])

    f,ax = plt.subplots(2,1, sharex=True)
    ax[0].semilogx(freqs, db(spice_volts), label="EAGLE")
    ax[0].semilogx(freqs, db( v ), label="4th order best fit")
    ax[1].plot(freqs, db(spice_volts)-db( v ))

    plt.legend()
    # plt.show()

def fit_stage2_decay(det=None):
    f, d1, d2 = read_second_stage()
    f_min_idx = 0
    f_max_idx = np.argmax(f> 1E6)
    freqs = f.values[f_min_idx:f_max_idx]
    spice_volts = d1[f_min_idx:f_max_idx]

    mod1 = DigitalFilter(1)
    # mod2 = DigitalFilter(2)
    mod1.num = [1,-1]
    # mod2.num = [1]

    if det is not None:
        det.AddDigitalFilter(mod1)

    def get_filter(mag):
        mag = 1 - 10**mag
        ws = freqs *(np.pi /  0.5E9)

        mod1.set_poles(mag)
        h1 = get_freqz(mod1, w = ws )

        return np.abs(h1)

    res = optimize.minimize(lsq, [-3], method="L-BFGS-B", args=(get_filter, spice_volts), bounds=[(-8,-1)])

    v = get_filter(*res["x"])

    f,ax = plt.subplots(2,1, sharex=True)
    ax[0].semilogx(freqs, db(spice_volts), label="EAGLE")
    ax[0].semilogx(freqs, db( v ), label="4th order best fit")
    ax[1].plot(freqs, db(spice_volts)-db( v ))

    plt.legend()
    # plt.show()

def fit_first_stage(det=None):
    f, d1 = read_first_stage()

    f_min_idx = np.argmax(f > 10**5)
    f_max_idx = np.argmax(f> 5E8)
    freqs = f.values[f_min_idx:f_max_idx]
    spice_volts = d1[f_min_idx:f_max_idx]

    mod1 = DigitalFilter(2)
    # mod2 = DigitalFilter(2)
    # mod1.num = [1]
    # mod2.num = [1]

    if det is not None:
        det.AddDigitalFilter(mod1)

    def get_filter(mag, zmag, phi):
        ws = freqs *(np.pi /  0.5E9)

        mag = 1-10**mag
        # mag2 = 1-10**mag2
        zmag = 1-10**zmag
        # zmag2 = 1-10**zmag2

        # mod1.set_zeros(zmag, zphi)
        mod1.num=[1,-zmag]
        mod1.set_poles(mag, phi)

        # mod2.set_zeros(zmag2)
        # mod2.set_poles(zmag, phi2)

        h1 = get_freqz(mod1, w = ws )
        h2=1
        # h2 = get_freqz(mod2, w = ws )

        return np.abs(h1*h2)

    res = optimize.minimize(lsq, [-2, 0.9,  0.1], method="L-BFGS-B", args=(get_filter, spice_volts), bounds=[(-8,0), (0,1), (0, np.pi)])

    v = get_filter(*res["x"])
    print(res["x"])

    f,ax = plt.subplots(2,1, sharex=True)
    ax[0].semilogx(freqs, db(spice_volts), label="EAGLE")
    ax[0].semilogx(freqs, db( v ), label="3rd order filter")
    ax[1].plot(freqs, db(spice_volts)-db( v ))

    ax[0].legend()
    # plt.show()

def fit_second_stage_lowpass(det=None):
    f, d1, d2 = read_second_stage()

    f_min_idx = np.argmax(f > 10**6)
    f_max_idx = np.argmax(f> 5E8)
    freqs = f.values[f_min_idx:f_max_idx]
    spice_volts = d1[f_min_idx:f_max_idx]

    mod1 = DigitalFilter(2)
    mod2 = DigitalFilter(1)
    # mod1.num = [1,1,0]
    # mod2.num = [1]

    if det is not None:
        det.AddDigitalFilter(mod1)
        # det.AddDigitalFilter(mod2)

    def get_filter(mag, phi, zmag, zphi,):
        ws = freqs *(np.pi /  0.5E9)

        mag = 1-10**mag
        zmag = 1-10**zmag
        mod1.set_zeros(zmag, zphi)
        # mod1.num=[1,0.]
        mod1.set_poles(mag, phi )

        # mod2.set_zeros(zmag2)
        # mag2 = 1-10**mag2
        # zmag2 = 1-10**zmag2
        # mod2.set_poles(mag2)

        h1 = get_freqz(mod1, w = ws )
        # h2 = get_freqz(mod2, w = ws )
        h2=1

        return np.abs(h1*h2)

    res = optimize.minimize(lsq, [-2, 0.1, -2, 0.1], method="L-BFGS-B", args=(get_filter, spice_volts), bounds=[(-8,0), (0, np.pi), (-8,0), (0, np.pi)])

    v = get_filter(*res["x"])
    print(res["x"])
    print(mod1.num)

    f,ax = plt.subplots(2,1, sharex=True)
    ax[0].semilogx(freqs, db(spice_volts), label="EAGLE")
    ax[0].semilogx(freqs, db( v ), label="3rd order filter")
    ax[1].plot(freqs, db(spice_volts)-db( v ))

    ax[0].legend()
    # plt.show()

def read_digitzer():
    file_name = "digitizer_front_end.txt"
    df = pd.read_table(file_name, header=0, sep="\s+")
    f = df["FREQ"]
    return f, df["Vmain"]/df["Vmain"].max(), df["VMain2"]/df["VMain2"].max()

def read_first_stage():
    file_name = "total_preamp_1k_to_1g.txt"
    df = pd.read_table(file_name, header=0, sep="\s+")
    f = df["FREQ"]
    return (f, df["VM(FIRST)"]/df["VM(FIRST)"].max())

def read_second_stage():
    file_name = "second_stage_1k_to_1g.txt"
    df = pd.read_table(file_name, header=0, sep="\s+")
    f = df["FREQ"]

    return (f, df["VM(OUT2_HI_N)"]/df["VM(OUT2_HI_N)"][np.argmax(f>=10**6)], df["VM(OUT2_HI_P)"]/df["VM(OUT2_HI_P)"][np.argmax(f>=10**6)])

def db(voltage):
    return 10*np.log10(voltage)

def lsq(x, filter_func, spice_volts):
    v = filter_func(*x)
    return np.sum((spice_volts-v)**2)

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
