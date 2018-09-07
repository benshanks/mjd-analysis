#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import pygama.decoders as dl
from pygama.transforms import rc_decay
from pygama.waveform import Waveform
import matplotlib.pyplot as plt
from scipy.signal import medfilt
plt.style.use('presentation')


def main():

    chan = 32
    file_name = "pulser_data/chan{}_pulsers.npz".format(chan)

    wf_file = np.load(file_name)
    wfs = wf_file["wfs"]

    for wf in wfs[:1]:
        wf_corr = wf.data - np.mean(wf.data[:500])
        wf_corr /= np.mean(wf_corr[-100:])
        mid_idx = np.argmax(wf_corr>0.1)
        wf_window = wf_corr[mid_idx-100:]
        plt.plot(np.arange(len(wf_window))*10/1000-1, medfilt(wf_window, kernel_size=11))

    plt.xlabel("Time [µs]")
    plt.ylabel("ADC [normalized]")

    plt.xlim(-0.5, 12.5)
    plt.ylim(0.975, 1.025)
    plt.tight_layout()
    plt.savefig("overshoot.pdf")
    plt.show()

if __name__=="__main__":
    main()
