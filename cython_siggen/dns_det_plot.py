#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib import gridspec

import numpy as np

import helpers
from detector_model import *
from probability_model_hdxwf import *
from probability_model_waveform import *


#from dns_wf_model import *

fitSamples = 425
timeStepSize = 1

wfFileName = "P42574A_12_fastandslow_oldwfs.npz"
if os.path.isfile(wfFileName):
  data = np.load(wfFileName)
  wfs = data['wfs']
  results = data['results']
  wfs = wfs[:10]
  results = results[:10]

  #i think wfs 1 and 3 might be MSE
  wfs = np.delete(wfs, [1,3])
  results = np.delete(results, [1,3])

  numWaveforms = wfs.size
  print "Fitting %d waveforms" % numWaveforms

else:
  print "No saved waveforms available.  Loading from Data"
  exit(0)


tempGuess = 79.071172
gradGuess = 0.04
pcRadGuess = 2.5
pcLenGuess = 1.6
#Create a detector model
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)
det.LoadFieldsGrad("fields_impgrad_0-0.06.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)

tf_first_idx = 0
velo_first_idx = 6
trap_idx = 12
grad_idx = 13

colors = ["red" ,"blue", "green", "purple", "cyan", "magenta", "goldenrod", "brown" ]

def main(argv):

  fig1 = plt.figure(1, figsize=(20,10))
  plt.clf()
  gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")

  for wf in wfs:
      wf.WindowWaveformTimepoint(fallPercentage=.97, rmsMult=2, earlySamples=100)
      dataLen = wf.wfLength
      t_data = np.arange(dataLen) * 10
      ax0.plot(t_data, wf.windowedWf, color="black")

#  data = np.loadtxt("posterior_sample.txt")
  data = np.loadtxt("sample.txt")

  print "found %d samples" % len(data)

  for (idx,params) in enumerate(data[-20:]):

    b_over_a, c, d, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
    h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
    charge_trapping = params[trap_idx]
    grad = np.int(params[grad_idx])

    det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
    det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
    det.trapping_rc = charge_trapping
    det.SetFieldsGradIdx(grad)

    rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[14:].reshape((8, numWaveforms))
    print "sample %d:" % idx
    print "  tf params: ",
    print b_over_a, c, d, rc1, rc2, rcfrac
    print "  velo params: ",
    print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0
    print "  charge trapping: ",
    print params[trap_idx]
    print "  grad idx (grad): ",
    print params[grad_idx],
    print " (%0.3f)" % det.gradList[grad]

    for (wf_idx,wf) in enumerate(wfs):
      rad, phi, theta = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx]
      scale, t0, smooth =  scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx]
      m, b = m_arr[wf_idx], b_arr[wf_idx]
      r = rad#rad * np.cos(theta)
      z = theta#rad * np.sin(theta)
      print "  wf number %d:" % wf_idx
      print "    r: %0.2f , phi: %0.4f, z:%0.2f" % (r, phi/np.pi, z)
      print "    rad: %0.2f, theta: %0.4f" % (rad, theta/np.pi)
      print "    t0: %0.2f" % t0
      print "    m: %0.3f, b: %0.3f" % (m,b)

      ml_wf = det.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)
      if ml_wf is None:
        continue

      baseline_trend = np.linspace(b, m*fitSamples+b, fitSamples)
      ml_wf += baseline_trend

      dataLen = wf.wfLength
      t_data = np.arange(dataLen) * 10
      ax0.plot(t_data, ml_wf[:dataLen], color=colors[wf_idx], alpha=0.1)
      ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color=colors[wf_idx],alpha=0.1)

#  ax0.set_ylim(0, wf.wfMax*1.1)
  ax1.set_ylim(-50, 50)
  plt.show()


if __name__=="__main__":
    main(sys.argv[1:])