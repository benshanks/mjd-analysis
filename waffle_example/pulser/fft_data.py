#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import pygama.decoders as dl
from pygama.transforms import rc_decay
from pygama.waveform import Waveform
from scipy import signal


def main():
    plot_pulser_ffts("chan632_8wfs.npz")

def plot_pulser_ffts(wfFileName):

    if os.path.isfile(wfFileName):
        print("Loading wf file {0}".format(wfFileName))
        data = np.load(wfFileName, encoding="latin1")
        wfs = data['wfs']

    avg_count = 0
    for wf in wfs:
        wf.window_waveform(time_point=0.95, early_samples=125, num_samples=1000)

        wf_data = wf.windowed_wf / wf.amplitude
        # plt.plot(wf_data)
        top_idx = 140#np.argmax( wf_data > 1000) + 15
        wf_top = wf_data[ top_idx:top_idx+800 ]

        xf,power = signal.periodogram(wf_top, fs=1E8, detrend="linear")
        plt.semilogx(xf,power)

        #
        #
        #
        # avg_wf += wf_top
        # avg_count +=1
        #
        # if avg_count >= pulser_count: break
    plt.xlim(10**6, 0.5*10**8)
    plt.show()

    #     xf,power = signal.periodogram(avg_wf, fs=1E8, detrend="linear")
    #     plt.semilogx(xf,power, label="channel {}".format(chan))
    # plt.legend(loc=2)
    # inp = input("q to quit, else to continue")
    # if inp == "q": exit()



if __name__=="__main__":
    main()
