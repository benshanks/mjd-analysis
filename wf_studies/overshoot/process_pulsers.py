#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import pygama.decoders as dl
from pygama.transforms import rc_decay
from pygama.waveform import Waveform


def main():
    # process()
    save_pulsers()
    # plot_pulser_ffts()


def process():

    runList = np.arange(848, 849)

    chanList = None

    process_tier_0("", runList, output_dir="")

def save_pulsers(num_to_save = 10):
    run_number = 848

    file_name = "t1_run{}.h5".format(run_number)
    df_gretina = pd.read_hdf(file_name, key="ORGretina4MWaveformDecoder")

    g4 = dl.Gretina4MDecoder(file_name)

    df_gretina = df_gretina[df_gretina.energy > 0.5E9]
    channels = df_gretina.channel.unique()

    plt.ion()
    plt.figure()

    for chan in channels:
        plt.clf()
        plt.xlabel("Time [ns]")
        plt.ylabel("ADC [arb]")
        plt.title("Channel {}".format(chan))

        df_chan = df_gretina[df_gretina.channel == chan]

        chan_wfs = []

        for i, (index, row) in enumerate(df_chan.iterrows()):
            if index < 10: continue
            wf = g4.parse_event_data(row)
            wf.bl_int = np.mean(wf.data[:200])
            wf.bl_slope=0
            mean_sub_data = wf.data - wf.bl_int

            if np.amax(mean_sub_data) < 5000: continue
            if np.count_nonzero( mean_sub_data > 0.5*mean_sub_data.max()) < 10: continue

            # plt.plot(mean_sub_data)
            wf_window = wf.window_waveform(1000, 10, 100, method="value")
            chan_wfs.append(  wf )
            plt.plot(wf_window, c="b", alpha=0.2 )

            if len(chan_wfs) >= num_to_save: break

        if len(chan_wfs) < num_to_save: continue
        plt.savefig("pulser_data/chan{}_pulsers.png".format(chan))
        np.savez("pulser_data/chan{}_pulsers.npz".format(chan), wfs=chan_wfs)

        # inp = input("q to quit, else to continue")
        # if inp == "q": exit()

def plot_pulser_ffts():
    run_number = 845

    proc = DataProcessor()

    file_name = "t1_run{}.h5".format(run_number)
    df_gretina = pd.read_hdf(file_name, key="ORGretina4MWaveformDecoder")

    g4 = dl.Gretina4MDecoder(file_name)

    df_gretina = df_gretina[df_gretina.energy > 0.5E9]
    channels = df_gretina.channel.unique()

    plt.ion()
    plt.figure(figsize=(14,7))
    ax1 = plt.gca()
    ax1.set_xlim(10**6, 0.5*10**8)

    pulser_count = 50

    for chan in channels:

        df_chan = df_gretina[df_gretina.channel == chan]
        avg_wf = np.zeros(900)
        avg_count = 0
        for i, (index, row) in enumerate(df_chan.iterrows()):
            wf = g4.parse_event_data(row)
            mean_sub_data = wf.data - np.mean(wf.data[:750])
            if np.amax(mean_sub_data) < 200: continue
            if np.count_nonzero( mean_sub_data > 0.5*mean_sub_data.max()) < 10: continue

            top_idx = np.argmax( mean_sub_data > 1000) + 4
            wf_top = mean_sub_data[ top_idx:top_idx+900 ]

            avg_wf += wf_top
            avg_count +=1

            if avg_count >= pulser_count: break
        if avg_count < pulser_count: continue
        avg_wf/=pulser_count


        # plt.plot(avg_wf)
        xf,power = signal.periodogram(avg_wf, fs=1E8, detrend="linear")
        plt.semilogx(xf[xf>1E6],power[xf>1E6], label="channel {}".format(chan))
    plt.legend(loc=2)
    inp = input("q to quit, else to continue")
    if inp == "q": exit()

def square_model(energy, t0, decay_time, decay_frac, length):

    square = np.zeros(length)
    square[t0:] = energy

    num, den = rc_decay(decay_time)

    square = square + decay_frac*signal.lfilter(num, den, square)
    return square


if __name__=="__main__":
    main()
