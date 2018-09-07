#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4
import pandas as pd

from waffle.processing import DataProcessor
from waffle.pz_fitter import *

def main(chan, doPlot=False):
    chan = int(chan)

    #Load the PZ params from the fit
    proc = DataProcessor(None)
    df_pz = pd.read_hdf(proc.channel_info_file_name, key="pz")
    pz_chan = df_pz.loc[df_pz.channel==chan]

    rc_us = pz_chan.rc_us.values
    rc_ms = 1E3*pz_chan.rc_ms.values

    fit_us = 72.07
    fit_ms = 1.329

    # runList = np.arange(11510, 11600)
    runList = np.arange(11510, 11530)
    df = proc.load_t2(runList)
    df_chan = df[df.channel == chan]

    df_bl = pd.read_hdf(proc.channel_info_file_name, key="baseline")
    bl_cuts = df_bl.loc[chan]
    bl_norm = bl_cuts.bl_int_mean

    #drop nan's, i.e., the first event in every run
    df_chan_nona = df_chan.dropna()
    cut = (df_chan_nona.prev_e > 500) & (df_chan_nona.ecal > 100) & (df_chan_nona.isPulser==0)
    df_cut = df_chan_nona[cut]

    #make sure the previous event is at an OK baseline
    df_good = df_cut[ (df_cut.prev_bl > bl_cuts.bl_int_min) & (df_cut.prev_bl < bl_cuts.bl_int_max)   ]

    bl_good = df_good["bl_int"]
    prev_amp = df_good["prev_amp"]
    previous_times = df_good["prev_t"]

    bl_dev = (bl_good-bl_norm)/prev_amp

    time_ms = df_good["prev_t"]*10/1E6
    ms_max = 50
    cut = (time_ms<ms_max) & (time_ms>0.05) & (bl_dev > -0.05)
    time_ms = time_ms[cut]
    bl_dev=bl_dev[cut]

    rc_ms, rc_us, offset, toffset = fit_data(time_ms, bl_dev)

    f = plt.figure()
    plot_data(time_ms, bl_dev, rc_ms, rc_us, offset, toffset)

    t_plot = np.arange(0, np.amax(time_ms), 0.001)
    plt.plot(t_plot, get_decay(t_plot, fit_ms, fit_us, offset, toffset), label="Full fit params")
    plt.plot(t_plot, get_decay(t_plot, 1.3458514126441128, 74.48390773033135, offset, toffset), label="Pulser fit params")

    plt.legend()

    plt.show()





if __name__=="__main__":
    main(*sys.argv[1:])
