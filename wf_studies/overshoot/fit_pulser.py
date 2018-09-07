#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4

from waffle.management import LocalFitManager, FitConfiguration

def main(chan, doPlot=False):

    chan = int(chan)

    wf_idxs = np.arange(0,2)
    directory = "{}pulsers_test_chan{}".format(len(wf_idxs), chan)
    wf_file = "pulser_data/chan{}_pulsers.npz".format(chan)

    wf_conf = {
        "wf_file_name":wf_file,
        "wf_idxs":wf_idxs,
        "align_idx":5,
        "num_samples":500,
        "align_percent":5000 #ADC value rather than percentage for pulsers
    }

    model_conf = [
        ("AntialiasingFilterModel",  {}),

        # ("AntialiasingFilterModel",  {"pmag_lims":[0.6, 0.7], "pphi_lims":[0.05, 0.15]}),
        ("OscillationFilterModel",  {"pphi_lims":[0.1,50]}), #preamp oscillation
        # ("OscillationFilterModel",  {"pphi_lims":[15,50]}), #preamp oscillation
        # ("DigitalFilterModel",  {"order":2, "include_zeros":True}),
        # ("DigitalFilterModel",  {"order":2, "include_zeros":False}),
        ("OvershootFilterModel",{"zmag_lims":[1, 3]})
        # ("DigitalFilterModel",  {"order":1, "include_zeros":True, "pmag_lims":[-15, 0], "zmag_lims":[-15, 0], "pphi_lims":[-15,0], "zphi_lims":[-15,0], "exp":True}),
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
