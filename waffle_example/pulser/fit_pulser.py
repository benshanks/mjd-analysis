#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4
import pandas as pd

from waffle.management import LocalFitManager, FitConfiguration
from waffle.processing import DataProcessor

def main(chan, doPlot=False):

    chan = int(chan)

    wf_idxs = np.arange(0,2)
    directory = "{}pulsers_test_chan{}".format(len(wf_idxs), chan)
    wf_file = "pulser_data/chan{}_pulsers.npz".format(chan)

    wf_conf = {
        "wf_file_name":wf_file,
        "wf_idxs":wf_idxs,
        "align_idx":50,
        "num_samples":1000,
        "align_percent":50 #ADC value rather than percentage for pulsers
    }

    rc_antialias = 2 * 49.9 * 33E-12
    rc_antialias_hi = np.exp(-1./1E9/(2*rc_antialias))
    rc_antialias_lo = np.exp(-1./1E9/(0.5*rc_antialias))

    #Load the PZ params from the fit
    proc = DataProcessor(None)
    df_pz = pd.read_hdf(proc.channel_info_file_name, key="pz")
    pz_chan = df_pz.loc[df_pz.channel==chan]

    rc_us = pz_chan.rc_us.values
    rc_ms = 1E3*pz_chan.rc_ms.values

    model_conf = [
        #Preamp effects
        ("LowPassFilterModel",  {"order":2,"pmag_lims":[0.95,1], "pphi_lims":[0,0.01]}), #preamp?


        ("HiPassFilterModel",   {"order":1, "pmag_lims": [0.8*rc_us,1.2*rc_us]}), #rc decay filter (~70 us), second stage
        ("HiPassFilterModel",   {"order":1, "pmag_lims": [0.8*rc_ms,1.2*rc_ms]}),
        #Gretina card effects
        ("AntialiasingFilterModel",  {}), #antialiasing
        ("OvershootFilterModel",{"zmag_lims":[1, 3]}), #gretina overshoot
        #Shitty ringing
        # ("OscillationFilterModel",  {"include_zeros":True}), #preamp oscillation at 5 MHz
        ("OscillationFilterModel",  {"include_zeros":True, "pphi_lims":[8, 15] }), #preamp oscillation at 10 Mhz
        ("OscillationFilterModel",  {"include_zeros":True, "pphi_lims":[9, 11.0] }), #preamp oscillation at 10 Mhz
        ("OscillationFilterModel",  {"include_zeros":True, "pphi_lims":[11, 15.0] }), #preamp oscillation at 10 Mhz
        ]

    conf = FitConfiguration(
        "",
        directory = directory,
        wf_conf=wf_conf,
        model_conf=model_conf,
        joint_energy=True,
        joint_risetime=True,
        interpType="linear"
    )

    if os.path.isdir(directory):
        pass
        # if len(os.listdir(directory)) >0:
        #     raise OSError("Directory {} already exists: not gonna over-write it".format(directory))
    else:
        os.makedirs(directory)

    fm = LocalFitManager(conf, num_threads=1, model_type="PulserTrainingModel")

    conf.save_config()
    fm.fit(numLevels=1000, directory = directory,new_level_interval=5000, numParticles=3)


if __name__=="__main__":
    main(*sys.argv[1:])
