#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""

import dnest4

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, shutil
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm

# import pandas as pd
import numpy as np
from scipy import signal
import multiprocessing

import helpers
from pysiggen import Detector

from dns_tf_model import *

doInitPlot =0
# doWaveformPlot =0
# doHists = 1
# plotNum = 1000 #for plotting during the Run
doWaveformPlot =0
doHists = 1
plotNum = 1000 #for plotting during the Run
numThreads = multiprocessing.cpu_count()

max_sample_idx = 200
fallPercentage = 0.95
fieldFileName = "P42574A_fields_impgrad_0.00000-0.00100.npz"

wfFileName = "P42574A_24_spread.npz"
# wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    #i think wfs 1 and 3 might be MSE
    #wf 2 is super weird

    wfs = data['wfs']
    #
    # wfidxs = [0, 5, 8, 14]
    # wfs = wfs[wfidxs]

    # one slow waveform
    fitwfnum = 5
    wfs = wfs[:fitwfnum+1]
    wfs = np.delete(wfs, range(0,fitwfnum))

    numWaveforms = wfs.size
    print "Fitting %d waveforms" % numWaveforms,
    if numWaveforms < numThreads:
      numThreads = numWaveforms
    print "using %d threads" % numThreads

else:
  print "Saved waveform file %s not available" % wfFileName
  exit(0)

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey" ]

wfLengths = np.empty(numWaveforms)
wfMaxes = np.empty(numWaveforms)


if doInitPlot: plt.figure(500)
baselineLengths = np.empty(numWaveforms)
for (wf_idx,wf) in enumerate(wfs):
  wf.WindowWaveformAroundMax(fallPercentage=fallPercentage, rmsMult=2, earlySamples=max_sample_idx)
  baselineLengths[wf_idx] = wf.t0Guess

  print "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber)
  wfLengths[wf_idx] = wf.wfLength
  wfMaxes[wf_idx] = np.argmax(wf.windowedWf)

  if doInitPlot:
      if len(colors) < numWaveforms:
          color = "red"
      else: color = colors[wf_idx]
      plt.plot(wf.windowedWf, color=color)

baseline_origin_idx = np.amin(baselineLengths) - 30
if baseline_origin_idx < 0:
    print "not enough baseline!!"
    exit(0)

initT0Padding(max_sample_idx, baseline_origin_idx)

if doInitPlot:
    plt.show()
    exit()

siggen_wf_length = (max_sample_idx - np.amin(baselineLengths) + 10)*10
output_wf_length = np.amax(wfLengths) + 1

#Create a detector model
timeStepSize = 1 #ns
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, timeStep=timeStepSize, numSteps=siggen_wf_length, maxWfOutputLength =output_wf_length, t0_padding=100 )
det.LoadFieldsGrad(fieldFileName)
det.SetFieldsGradIdx(10)

def fit(directory):

  initializeDetectorAndWaveforms(det, wfs,)
  initMultiThreading(numThreads)

  # Create a model object and a sampler
  model = Model()
  sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                    sep=" "))

  # Set up the sampler. The first argument is max_num_levels
  gen = sampler.sample(max_num_levels=150, num_steps=100000, new_level_interval=10000,
                        num_per_step=1000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=1234)

  # Do the sampling (one iteration here = one particle save)
  for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

  # Run the postprocessing
  # dnest4.postprocess()



def plot(sample_file_name, directory):
    fig1 = plt.figure(0, figsize=(20,10))
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")

    for wf in wfs:
      dataLen = wf.wfLength
      t_data = np.arange(dataLen) * 10
      ax0.plot(t_data, wf.windowedWf, color="black")

    sample_file_name = directory + sample_file_name
    if sample_file_name == directory + "sample.txt":
      shutil.copy(directory+ "sample.txt", directory+"sample_plot.txt")
      sample_file_name = directory + "sample_plot.txt"

    data = np.loadtxt( sample_file_name)
    num_samples = len(data)

    # data = pd.read_csv("sample_plot.txt", delim_whitespace=True, header=None)
    # num_samples = len(data.index)
    print "found %d samples" % num_samples

    print sample_file_name

    if sample_file_name== (directory+"sample_plot.txt"):
        if num_samples > plotNum: num_samples = plotNum
    print "plotting %d samples" % num_samples
    # exit(0)

    r_arr = np.empty((numWaveforms, num_samples))
    z_arr = np.empty((numWaveforms, num_samples))
    tf = np.empty((6, num_samples))
    wf_params = np.empty((numWaveforms, 8, num_samples))

    for (idx,params) in enumerate(data[-num_samples:]):
        tf_b, gain, d2, rc1, rc2, rcfrac = params[0:6]
        tf[:,idx] = tf_b, gain, d2, rc1, rc2, rcfrac

        tf_2c = gain - 1 - d2

        det.SetTransferFunction(tf_b, tf_2c, d2, rc1, rc2, rcfrac, isDirect=True)

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[6:].reshape((8, numWaveforms))
        print "sample %d:" % idx
        print "  tf params: ",
        print params[0:6]

        for (wf_idx,wf) in enumerate(wfs):
          r, phi, z = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
          scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
          m, b = m_arr[wf_idx], b_arr[wf_idx]
          wf_params[wf_idx, :, idx] = r, phi, z, scale, t0, smooth, m, b

          r_arr[wf_idx, idx], z_arr[wf_idx, idx] = r,z

          if doWaveformPlot:
            ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  np.int(output_wf_length), h_smoothing = smooth, alignPoint="max")
            if ml_wf is None:
                continue

            start_idx = -baseline_origin_idx
            end_idx = output_wf_length - baseline_origin_idx - 1
            baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, output_wf_length)
            ml_wf += baseline_trend

            dataLen = wf.wfLength
            t_data = np.arange(dataLen) * 10
            ax0.plot(t_data, ml_wf[:dataLen], color=colors[wf_idx], alpha=0.1)
            ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color=colors[wf_idx],alpha=0.1)

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)

    if not doHists:
        plt.show()
        exit()

    vFig = plt.figure(2, figsize=(20,10))
    tfLabels = ['b_ov_a', 'c', 'd', 'rc1', 'rc2', 'rcfrac']
    vLabels = ['h_100_mu0', 'h_100_beta', 'h_100_e0','h_111_mu0','h_111_beta', 'h_111_e0']
    vmodes, tfmodes = np.empty(6), np.empty(6)
    num_bins = 100
    for i in range(6):
        axis = vFig.add_subplot(6,1,i+1)
        axis.set_ylabel(tfLabels[i])
        [n, b, p] = axis.hist(tf[i,:], bins=num_bins)
        max_idx = np.argmax(n)
        print "%s mode: %f" % (tfLabels[i], b[max_idx])

    positionFig = plt.figure(3, figsize=(15,15))
    plt.clf()
    colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "PuRd"]

    for wf_idx in range(numWaveforms):
        xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*10+1)
        yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*10+1)
        plt.hist2d(r_arr[wf_idx,:], z_arr[wf_idx,:],  bins=[ xedges,yedges  ], norm=LogNorm(), cmap=plt.get_cmap(colorbars[wf_idx]))
        rad_mean = np.mean(wf_params[wf_idx, 0,:])
        print "wf %d rad: %f + %f - %f" % (wf_idx, rad_mean, np.percentile(wf_params[wf_idx, 0,:], 84.1)-rad_mean, rad_mean- np.percentile(wf_params[wf_idx, 0,:], 15.9) )
        # plt.colorbar()
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")
    plt.axis('equal')

    if numWaveforms == 1:
        #TODO: make this plot work for a bunch of wfs
        vFig = plt.figure(4, figsize=(20,10))
        wfLabels = ['rad', 'phi', 'theta', 'scale', 't0', 'smooth', 'm', 'b']
        num_bins = 100
        for i in range(8):
            axis = vFig.add_subplot(4,2,i+1)
            axis.set_ylabel(wfLabels[i])
            [n, b, p] = axis.hist(wf_params[0, i,:], bins=num_bins)
            # if i == 4:
            #     axis.axvline(x=t0_min, color="r")
            #     axis.axvline(x=t0_max, color="r")
            #     axis.axvline(x=t0_guess, color="g")


    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        fit("")
    if len(sys.argv) >= 3:
        directory = sys.argv[2]
    else:
        directory = ""

    if sys.argv[1] == "plot":
        plot("sample.txt", directory)
    elif sys.argv[1] == "plot_post":
        plot("posterior_sample.txt", directory)
    else:
        fit(sys.argv[1])
