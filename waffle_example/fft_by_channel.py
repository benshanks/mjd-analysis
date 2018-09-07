#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import pygama.filters as filt
import os
import matplotlib.pyplot as plt
import pygama.decoders as dl
from scipy import optimize

def main():

    runList = np.arange(11515, 11516)
    proc = DataProcessor()


    plt.ion()
    f = plt.figure(figsize=(12,9))

    rc_num, rc_den = filt.rc_decay(72)
    overshoot_num, overshoot_den = filt.gretina_overshoot(2, -3.5)

    # print(overshoot_num, overshoot_den)
    # exit()

    ds_inf = pd.read_csv("ds1_run_info.csv")

    for runNumber in runList:
        t1_file = os.path.join(proc.t1_data_dir,  "t1_run{}.h5".format(runNumber))
        df = pd.read_hdf(t1_file,key="ORGretina4MWaveformDecoder")
        g4 = dl.Gretina4MDecoder(t1_file)

        chanList = np.unique(df["channel"])
        chanList = [ 580]

        pz_fun = []
        masses = []
        resolutions = []

        for chan in chanList:
            print ("Channel {}".format(chan))

            try:
                ds_inf_det = ds_inf[(ds_inf.LG==chan) | (ds_inf.HG==chan) ]
                det_mass = ds_inf_det.iloc[0].Mass
                det_res = ds_inf_det.iloc[0].Resolution
                det_ctres = ds_inf_det.iloc[0].ct_resolution
                trap_factor = (det_res-det_ctres)/det_ctres
            except IndexError:
                print ("...couldn't find channel info")

            if chan%2 == 1: continue
            if not is_mj(chan): continue

            plt.clf()
            ax1 = plt.subplot(2,2,2)
            ax2 = plt.subplot(2,2,4)
            ax3 = plt.subplot(2,2,3)
            ax4 = plt.subplot(2,2,1)

            df_chan = df[ (df.channel == chan) & (df.energy > 0.2E9) ]

            e_min = df_chan.energy.min()
            e_max = df_chan.energy.max()

            e_cut = 0.9*(e_max-e_min) + e_min

            df_cut = df_chan[df_chan.energy > e_cut]

            bl_idx = 800
            baseline = np.zeros(bl_idx)
            flat_top = np.zeros(800)

            # plt.figure()
            num_wfs = 0
            for i, (index, row) in enumerate(df_cut.iterrows()):
                wf = g4.parse_event_data(row)
                wf_dat = wf.data - np.mean(wf.data[:bl_idx])

                try:
                    wf_corr, energy, gof = fit_tail(wf_dat)

                    align_idx = np.argmax(wf_corr/energy>0.999)+20

                    flat_top_wf = wf_corr[align_idx:align_idx+800] - energy

                    baseline += wf_dat[:bl_idx]
                    flat_top += flat_top_wf
                    num_wfs +=1


                except ValueError as e:
                    print(e)
                    continue

                ax1.plot(wf_corr[:800], c="b", alpha=0.1 )
                ax2.plot(flat_top_wf, c="b", alpha=0.1 )
                ax4.plot(wf_corr[align_idx-400:align_idx+805]/energy, c="b", alpha=0.1 )

                # plt.plot(flat_top_wf, c="b", alpha=0.1)

            # if num_wfs < 5: continue

            flat_top /=num_wfs
            baseline /= num_wfs

            pz_fun.append( np.sum(flat_top**2) )
            masses.append(det_res)

            ax1.set_title("Baseline")
            ax1.plot(baseline, c="r")

            ax2.set_title("Decay (PZ corrected)")
            ax2.plot(flat_top, c="r")
            # plt.plot(flat_top, c="r")

            plt.title("Channel {} (mass {}, res {})".format(chan, det_mass, det_res))
            # ax1.plot(baseline, label="baseline")
            # ax1.plot(flat_top - np.mean(flat_top), label="flat top")
            xf,power = signal.periodogram(baseline, fs=1E8, detrend="linear", scaling="spectrum")

            x_idx = np.argmax(xf>0.2E7)
            ax3.semilogx(xf[x_idx:],power[x_idx:], label="baseline")
            max_pwr = power.max()
            # ax2.plot(flat_top)
            xf,power = signal.periodogram(flat_top, fs=1E8, detrend="constant", scaling="spectrum")
            x_idx = np.argmax(xf>0.2E7)
            ax3.semilogx(xf[x_idx:],power[x_idx:], label="flat top")

            ax3.legend()

            plt.savefig("fft_plots/channel{}_fft.png".format(chan))

            # inp = input("q to continue, else to quit")
            # if inp == "q": exit()
        plt.figure()
        plt.scatter(masses, pz_fun)
        inp = input("q to continue, else to quit")
        if inp == "q": exit()


def fit_tail(wf_data):
    '''
    try to fit out the best tail parameters to flatten the top
    '''

    max_idx = np.argmax(wf_data)


    def min_func(x):
        rc_decay, overshoot_decay, overshoot_pole_rel, energy = x
        # rc_decay, overshoot_decay, overshoot_pole_rel, energy, long_rc = x

        rc_num, rc_den = filt.rc_decay(rc_decay*10)
        wf_proc1 = signal.lfilter(rc_den, rc_num, wf_data)

        # long_rc_num, long_rc_den = filt.rc_decay(long_rc*1000)
        # wf_proc1 = signal.lfilter(long_rc_den, long_rc_num, wf_proc1)

        overshoot_num, overshoot_den = filt.gretina_overshoot(overshoot_decay, overshoot_pole_rel)
        wf_proc = signal.lfilter(overshoot_den, overshoot_num, wf_proc1)

        tail_data = wf_proc[max_idx:]
        flat_line = np.ones(len(tail_data))*energy*1000
        return np.sum((tail_data-flat_line)**2)

    wf_max = wf_data.max()/1000

    res = optimize.minimize(min_func, [7.2, 2, -4, wf_max], method="Powell")
    # print(res["x"])
    rc1, rc2, f, e = res["x"]

    rc1*=10
    rc_num, rc_den = filt.rc_decay(rc1)
    wf_proc1 = signal.lfilter(rc_den, rc_num, wf_data)

    # long_rc*=1000
    # long_rc_num, long_rc_den = filt.rc_decay(long_rc)
    # wf_proc1 = signal.lfilter(long_rc_den, long_rc_num, wf_proc1)

    overshoot_num, overshoot_den = filt.gretina_overshoot(rc2, f)
    wf_proc = signal.lfilter(overshoot_den, overshoot_num, wf_proc1)

    return wf_proc, (e*1000), res["fun"]

    # plt.figure()
    # plt.plot(wf_data)
    # plt.plot(wf_proc1)
    # plt.plot(wf_proc)


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

if __name__=="__main__":
    main()
