#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4
import pandas as pd

from waffle.management import LocalFitManager, FitConfiguration
from waffle.processing import DataProcessor

chan_dict = {
594: "P42662B",
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}

def main(chan, doPlot=False):

    # directory = "4wf_648_zero"
    # wf_file = "training_data/chan648_8wfs.npz"
    # conf_name = "P42664A.conf"

    chan = int(chan)
    wf_idxs = np.arange(0,8)
    # wf_idxs = [0,3]
    directory = "8wf_free_{}".format(chan)

    wf_file = "training_data/chan{}_8wfs.npz".format(chan)
    conf_name = "{}.conf".format( chan_dict[chan] )

    # wf_file = "16wf_set_chan{}.npz".format(chan)
    # wf_idxs = np.arange(0,16,4)
    # wf_idxs = [1,4,8,12]

    datadir= os.environ['DATADIR']
    conf_file = datadir +"siggen/config_files/" + conf_name

    wf_conf = {
        "wf_file_name":wf_file,
        "wf_idxs":wf_idxs,
        "align_idx":125,
        "num_samples":1000,
        "do_smooth":False
    }

    #Load the PZ params from the fit
    proc = DataProcessor(None)
    df_pz = pd.read_hdf(proc.channel_info_file_name, key="pz")
    pz_chan = df_pz.loc[df_pz.channel==chan]

    rc_us = pz_chan.rc_us.values
    rc_ms = 1E3*pz_chan.rc_ms.values

    #this is ch 594 from the pulser
    overshoot_pole_mag = -5.246669688365981
    overshoot_zero_mag = 2.6703616866699833

    model_conf = [
        #Preamp effects
        ("LowPassFilterModel",  {"order":2,"pmag_lims":[0.95,1], "pphi_lims":[0,0.01]}), #preamp?
        ("HiPassFilterModel",   {"order":1, "pmag_lims": [0.9*rc_us,1.1*rc_us]}), #rc decay filter (~70 us), second stage
        ("HiPassFilterModel",   {"order":1, "pmag_lims": [0.9*rc_ms,1.1*rc_ms]}),
        # ("HiPassFilterModel",   {"order":1, "pmag_lims": [0.99*rc_us,1.01*rc_us]}), #rc decay filter (~70 us), second stage
        # ("HiPassFilterModel",   {"order":1, "pmag_lims": [0.99*rc_ms,1.01*rc_ms]}),
        #Gretina card effects
        ("AntialiasingFilterModel",  {}), #antialiasing
        ("OvershootFilterModel",{}), #gretina overshoot

        # ("OvershootFilterModel",{"zmag_lims": [0.99*overshoot_zero_mag,1.01*overshoot_zero_mag], "pmag_lims": [1.01*overshoot_pole_mag, 0.99*overshoot_pole_mag]}), #gretina overshoot
        #Detector effects
        ("ImpurityModelEnds",   {}),
        ("TrappingModel",       {}),
        ("VelocityModel",       {"include_beta":True}),
    ]

    conf = FitConfiguration(
        conf_file,
        directory = directory,
        wf_conf=wf_conf,
        model_conf=model_conf
    )

    if doPlot:
        import matplotlib.pyplot as plt
        # conf.plot_training_set()
        fm = LocalFitManager(conf, num_threads=1)
        for wf in fm.model.wfs:
            plt.plot(wf.windowed_wf)
            print (wf.window_length)
        plt.show()
        exit()

    if os.path.isdir(directory):
        if len(os.listdir(directory)) >0:
            raise OSError("Directory {} already exists: not gonna over-write it".format(directory))
    else:
        os.makedirs(directory)

    fm = LocalFitManager(conf, num_threads=len(wf_idxs))

    conf.save_config()
    fm.fit(numLevels=1000, directory = directory,new_level_interval=5000, numParticles=3)


if __name__=="__main__":
    main(*sys.argv[1:])
