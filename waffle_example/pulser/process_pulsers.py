#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import pygama.decoders as dl
from pygama.transforms import rc_decay
from pygama.waveform import Waveform
from scipy import signal

mjdList = [
582,583,580, 581,578, 579,
692 ,693 ,648, 649 ,640, 641 ,
610, 610,608, 609, 664, 665,
#624, 625, 628, 629,688, 689, 694, 695, 614, 615,
672, 673,
632, 633,626, 627, 690, 691,
600, 601, 598, 599,594, 595, 592, 593,
]

def main():
    # process()
    # plot_logic_signals()
    # plot_pulser()
    save_pulsers()
    # plot_pulser_ffts()


def plot_pulser():
    from waffle.models import PulserGenerator
    from siggen.electronics import DigitalFilter

    pg = PulserGenerator(1000)

    lowpass = DigitalFilter(2)
    lowpass.num = [1,2,1]
    lowpass.set_poles(0.975, 0.007)

    hipass = DigitalFilter(1)
    hipass.num, hipass.den = rc_decay(82, 1E9)

    pg.digital_filters.append(lowpass)
    pg.digital_filters.append(hipass)

    pg.energy = 1000
    p = pg.make_pulser(50, 500, 250)

    plt.figure()
    plt.plot(p)
    plt.show()

def process():

    runList = np.arange(11499, 11501)

    chanList = None

    #data processing

    proc = DataProcessor(mjdList)
    proc.tier0(runList, chan_list=chanList)

def plot_logic_signals():
    run_number = 11499

    proc = DataProcessor(DataProcessor)

    file_name = os.path.join(proc.t1_data_dir, "t1_run{}.h5".format(run_number))
    df_gretina = pd.read_hdf(file_name, key="ORGretina4MWaveformDecoder")

    df_gretina = df_gretina[df_gretina.energy > 0.5E9]
    # plot the energy spectrum
    # plt.figure()
    # plt.hist(df_gretina.energy)
    # plt.show()
    # exit()

    plt.ion()
    plt.figure()

    channels = df_gretina.channel.unique()

    g4 = dl.Gretina4MDecoder(file_name)
    for chan in channels:
        if chan%2 == 1: continue
        if not is_mj(chan): continue

        # plt.clf()
        plt.xlabel("Time [ns]")
        plt.ylabel("ADC [arb]")
        plt.title("Channel {}".format(chan))

        df_chan = df_gretina[df_gretina.channel == chan]

        for i, (index, row) in enumerate(df_chan.iterrows()):
            wf = g4.parse_event_data(row)
            mean_sub_data = wf.data - np.mean(wf.data[:750])

            if np.amax(mean_sub_data) < 3000: continue

            plt.plot(wf.time, mean_sub_data, c="b", alpha=0.2 )

            square = square_model(mean_sub_data[-1], np.argmax(mean_sub_data > 3000), 1.7, 0.025, len(mean_sub_data))
            plt.plot(wf.time, square, c="r")

            if i > 25: break

        inp = input("q to quit, else to continue")
        if inp == "q": exit()

def save_pulsers(num_to_save = 10):
    run_number = 11499

    proc = DataProcessor(DataProcessor)

    file_name = os.path.join(proc.t1_data_dir, "t1_run{}.h5".format(run_number))
    df_gretina = pd.read_hdf(file_name, key="ORGretina4MWaveformDecoder")

    g4 = dl.Gretina4MDecoder(file_name)

    df_gretina = df_gretina[df_gretina.energy > 0.1E9]
    channels = df_gretina.channel.unique()

    plt.ion()
    plt.figure(figsize=(14,7))

    for chan in channels:
        if chan%2 == 1: continue
        if not is_mj(chan): continue

        plt.clf()
        ax0 = plt.subplot(1,2,1)

        plt.xlabel("Time [ns]")
        plt.ylabel("ADC [arb]")

        ax1 = plt.subplot(1,2,2)
        ax1.set_xlim(10**6, 0.5*10**8)
        plt.title("Channel {}".format(chan))

        df_chan = df_gretina[df_gretina.channel == chan]
        if len(df_chan) == 0:
            print("channel {} appears empty?".format(chan))
            continue

        chan_wfs = []

        for i, (index, row) in enumerate(df_chan.iterrows()):
            wf = g4.parse_event_data(row)
            mean_sub_data = wf.data - np.mean(wf.data[:750])
            # if np.amax(mean_sub_data) < 200: continue

            wf_window = wf.window_waveform(100, 50, 500, method="value")
            if len(wf_window) == 0: continue
            chan_wfs.append(  wf )

            ax0.plot(wf_window, c="b", alpha=0.2 )

            top_idx = np.argmax( mean_sub_data > 0.9 * mean_sub_data.max() )
            xf,power = signal.periodogram(mean_sub_data[ top_idx: ], fs=1E8, detrend="linear")
            ax1.plot(xf, power, alpha=0.5, color="b")

            if len(chan_wfs) >= num_to_save: break

        if len(chan_wfs) < num_to_save: continue
        plt.savefig("pulser_data/chan{}_pulsers.png".format(chan))
        np.savez("pulser_data/chan{}_pulsers.npz".format(chan), wfs=chan_wfs)

        # if chan==648: plt.show()
        #
        # inp = input("q to quit, else to continue")
        # if inp == "q": exit()

def plot_pulser_ffts():
    run_number = 11499

    proc = DataProcessor(DataProcessor)

    file_name = os.path.join(proc.t1_data_dir, "t1_run{}.h5".format(run_number))
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

        if chan%2 == 1: continue
        if not is_mj(chan): continue

        df_chan = df_gretina[df_gretina.channel == chan]
        avg_wf = np.zeros(900)
        avg_count = 0
        for i, (index, row) in enumerate(df_chan.iterrows()):
            wf = g4.parse_event_data(row)
            mean_sub_data = wf.data - np.mean(wf.data[:750])
            if np.amax(mean_sub_data) < 200: continue

            top_idx = np.argmax( mean_sub_data > 1000) + 15
            wf_top = mean_sub_data[ top_idx:top_idx+900 ]

            avg_wf += wf_top
            avg_count +=1

            if avg_count >= pulser_count: break
        if avg_count < pulser_count: continue
        avg_wf/=pulser_count


        xf,power = signal.periodogram(avg_wf, fs=1E8, detrend="linear")
        plt.semilogx(xf,power, label="channel {}".format(chan))
    plt.legend(loc=2)
    inp = input("q to quit, else to continue")
    if inp == "q": exit()

def is_mj(chan):
    mj_list = [
    584,585,582,583,580, 581 ,  578, 579,
    692 ,693 ,648, 649 ,640, 641 ,642, 643,
    616, 617, 610, 610,608, 609, 664, 665,
    624, 625, 628, 629,688, 689, 694, 695, 614, 615,
    680, 681, 678, 679,672, 673, 696, 697,
    632, 633, 630, 631,626, 627, 690, 691,
    600, 601, 598, 599,594, 595, 592, 593,
    ]

    return (chan in mj_list)

def square_model(energy, t0, decay_time, decay_frac, length):

    square = np.zeros(length)
    square[t0:] = energy

    num, den = rc_decay(decay_time)

    square = square + decay_frac*signal.lfilter(num, den, square)
    return square


if __name__=="__main__":
    main()
