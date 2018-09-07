#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import signal

from pygama.processing import process_tier_0
import pygama.decoders as dl
from pygama.transforms import rc_decay

def main():
    run_number = 845

    process(run_number)
    plot_logic_signals(run_number)

def process(run_number):
    runList = [run_number]
    process_tier_0("", runList, output_dir="")

def plot_logic_signals(run_number):
    file_name = "t1_run{}.h5".format(run_number)
    df_gretina = pd.read_hdf(file_name, key="ORGretina4MWaveformDecoder")

    plt.ion()
    plt.figure()

    channels = df_gretina.channel.unique()
    print(channels)

    g4 = dl.Gretina4MDecoder(file_name)
    for chan in channels:
        # plt.clf()
        plt.title("Channel {}".format(chan))

        df_chan = df_gretina[df_gretina.channel == chan]

        for i, (index, row) in enumerate(df_chan.iterrows()):
            wf = g4.parse_event_data(row)
            mean_sub_data = wf.data - np.mean(wf.data[:400])

            #max val
            if np.count_nonzero( mean_sub_data > 0.5*mean_sub_data.max()) < 10: continue

            align_data = align(mean_sub_data, 100)

            plt.plot(align_data, c="b", alpha=0.2 )

            square = square_model(align_data[-1], 98, 2, 2.75, 0.01, len(align_data))
            plt.plot(square, c="r")

            if i > 25: break

        inp = input("q to quit, else to continue")
        if inp == "q": exit()


def align(wf_data, before, after=9000):
    idx = np.argmax( wf_data > 0.5*wf_data[-1])
    return wf_data[idx-before:idx+after]


def square_model(energy, t0, rise_time, decay_time, decay_frac, length):

    square = np.zeros(length)
    square[t0:t0+rise_time] = np.linspace(0, energy, rise_time)
    square[t0+rise_time:] = energy

    rc_int = 2 * 49.9 * 33E-12
    # rc_int = rc_int_in_ns*1E-9
    rc_int_exp = np.exp(-1./1E8/rc_int)
    square = signal.lfilter([1,1]/(2/(1-rc_int_exp)), [1,-rc_int_exp], square)

    num, den = rc_decay(decay_time)
    square = square + decay_frac*signal.lfilter(num, den, square)

    return square


if __name__=="__main__":
    main()
