#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from waffle.plots import TrainingPlotter
from waffle.models import *
from waffle.models import PulserGenerator
def main():
    # plot_by_chan()
    # plot_by_gain()
    fit_results()

def fit_results():
    num_samples = 200
    chans = range(48,52)
    pg = PulserGenerator(200)
    pg.energy = 1000

    f, ax = plt.subplots(1,4,figsize=(14,7))
    f2, ax2 = plt.subplots(1,2,figsize=(14,7))

    colors = ["red", "blue", "green","purple"]

    for chanidx,chan in enumerate(chans):
        color = colors[chanidx]
        dir_name = "2pulsers_test_chan{}".format(chan)
        self = TrainingPlotter(dir_name, num_samples, model_type="PulserTrainingModel")
        ax2[0].loglog(np.nan, c=color, label="chan {}".format(chan))
        for mod in self.model.joint_models.models:
            if isinstance(mod, AntialiasingFilterModel):

                for i, data in self.plot_data.iterrows():
                    params = data[mod.start_idx:mod.start_idx+mod.num_params]
                    mod.apply_to_detector(params, pg)
                    w, h = mod.get_freqz(params, np.logspace(-5,1, 500, base=np.pi))
                    ax2[0].loglog(w,h, c=color, alpha=0.1,)

                    p = pg.make_pulser( 10, 100, 50)
                    ax2[1].plot(p, c=color, alpha=0.1)


        for i in range(self.model.num_det_params):
            tf_data = self.plot_data[i]

            #corresponding model
            model_idx = self.model.joint_models.index_map[i]
            param_model = self.model.joint_models.models[model_idx]

            model_param_idx = i - param_model.start_idx
            param = param_model.params[model_param_idx]

            h,b = np.histogram(tf_data,bins="auto")

            if  isinstance(param_model, OvershootFilterModel):
                ax[model_param_idx].plot(b[:-1],h,  ls="steps", color=color, label="chan {}".format(chan))
            elif isinstance(param_model, AntialiasingFilterModel):
                ax[2+model_param_idx].plot(b[:-1],h,  ls="steps", color=color, label="chan {}".format(chan))

            else: continue


    ax[0].legend()
    ax2[0].legend()
    plt.show()


def plot_by_chan():
    for filename in glob.iglob('pulser_data/*.npz'):
        chan = np.int( filename.split("/")[1][4:6] )
        color= "r" if chan%2 ==0  else "b"
        # print(filename.split("/")[1][4:6])
        p = None
        data = np.load(filename)
        wfs = data['wfs']
        for wf in wfs:
            wf_blr = (wf.data - np.mean(wf.data[:660]) )
            wf_norm = wf_blr/np.mean(wf_blr[-5:])
            if p is None:
                p = plt.plot(wf_norm, alpha=0.1, c=color   )
            else:
                plt.plot(wf_norm, c=p[0].get_color(), alpha=0.1)


    plt.show()

def plot_by_gain():

    low_wf = np.zeros(2018)
    hi_wf = np.zeros_like(low_wf)
    hi = 0
    lo=0

    for filename in glob.iglob('pulser_data/*.npz'):
        chan = np.int( filename.split("/")[1][4:6] )

        data = np.load(filename)
        wfs = data['wfs']
        for wf in wfs:
            wf_blr = (wf.data - np.mean(wf.data[:660]) )
            wf_norm = wf_blr/np.mean(wf_blr[-5:])

            if chan%2==0:
                hi_wf += wf_norm
                hi+=1
            else:
                low_wf += wf_norm
                lo+=1

    plt.plot(low_wf/lo, c="b", label="odd chan avg")
    plt.plot(hi_wf/hi, c="r", label="even chan avg")
    plt.legend()
    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )
