#!/usr/local/bin/python
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import ndimage


#funnyEntries = [4110, 66905]

class Waveform:
  def __init__(self, waveform_data, channel_number, run_number, entry_number, baseline_rms, energy=None, timeSinceLast=None, energyLast=None):
    self.waveformData = waveform_data
    self.channel = channel_number
    self.runNumber = run_number
    self.entry_number = entry_number
    self.baselineRMS = baseline_rms
    self.timeSinceLast = timeSinceLast
    self.energyLast = energyLast
    self.energy = energy


  def WindowWaveform(self, numSamples, earlySamples=20, t0riseTime = 0.005, rmsMult=1):
    '''Windows to a given number of samples'''
    self.wfMax = np.amax(self.waveformData)

    startGuess = self.EstimateT0(rmsMult)
    firstFitSampleIdx = startGuess-earlySamples
    lastFitSampleIdx = firstFitSampleIdx + numSamples

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = earlySamples
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def WindowWaveformTimepoint(self, earlySamples=20, fallPercentage=None, rmsMult=1):
    '''Does "smart" windowing by guessing t0 and wf max'''
    self.wfMax = np.amax(self.waveformData)

    startGuess = self.EstimateT0(rmsMult)
    firstFitSampleIdx = startGuess-earlySamples

    lastFitSampleIdx = self.EstimateFromMax(fallPercentage)

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = earlySamples
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def WindowWaveformAroundMax(self, earlySamples=200, fallPercentage=None, rmsMult=1):
    '''Does "smart" windowing by guessing t0 and wf max'''
    self.wfMaxIdx = np.argmax(self.waveformData)
    self.wfMax =    np.amax(self.waveformData)
    t0Guess = self.EstimateT0(rmsMult)
    firstFitSampleIdx = self.wfMaxIdx-earlySamples
    lastFitSampleIdx = self.EstimateFromMax(fallPercentage)

    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = t0Guess - firstFitSampleIdx
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def EstimateT0(self, rmsMult=1):
    smoothed_wf = ndimage.filters.gaussian_filter1d(self.waveformData, 2, )

    return np.where(np.less(smoothed_wf, rmsMult*self.baselineRMS))[0][-1]

  def EstimateFromMax(self, fallPercentage=None):

    if fallPercentage is None:
      searchValue =  self.wfMax - self.baselineRMS
    else:
      searchValue = fallPercentage * self.wfMax
    return np.where(np.greater(self.waveformData, searchValue))[0][-1]


########################################################################

def plotResidual(simWFArray, dataWF, figure=None, residAlpha=0.1):
  '''I'd be willing to hear the argument this shouldn't be in here so that i don't need to load matplotlib to run this module,
     but for now, i don't think it matters
  '''
  if figure is None:
    figure = plt.figure(figsize=(20,10))
  else:
    plt.figure(figure.number)
    plt.clf()

  gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")

  dataLen = len(dataWF)
  t_data = np.arange(len(dataWF)) * 10

  ax0.plot(t_data, dataWF  ,color="red", lw=2, alpha=0.8)

  for idx in range(simWFArray.shape[0]):
    simWF = simWFArray[idx,:dataLen]
    diff = simWF - dataWF

    ax0.plot(t_data, simWF  ,color="black", alpha = residAlpha  )
    ax1.plot(t_data, diff  ,color="#7BAFD4",  alpha = residAlpha )

  legend_line_1 = ax0.plot( np.NaN, np.NaN, color='r', label='Data (unfiltered)' )
  legend_line_2 = ax0.plot( np.NaN, np.NaN, color='black', label='Fit waveform' )

  first_legend = ax0.legend(loc=4)
  ax1.set_ylim(-20,20)

#  ax0.set_ylim(6200,6500)

########################################################################

def plotManyResidual(simWFArray, dataWFArray, figure=None, residAlpha=0.1):
  '''I'd be willing to hear the argument this shouldn't be in here so that i don't need to load matplotlib to run this module,
     but for now, i don't think it matters
  '''
  if figure is None:
    figure = plt.figure()
  else:
    plt.figure(figure.number)
    plt.clf()

  gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1], sharex=ax0)
  ax1.set_xlabel("Digitizer Time [ns]")
  ax0.set_ylabel("Voltage [Arb.]")
  ax1.set_ylabel("Residual")

  simwfnum = simWFArray.shape[0]
  colors = ["red", "blue" , "green", "orange", "purple", "black", "cyan"]

  for (dataIdx, dataWFObj) in enumerate(dataWFArray):

    dataLen = dataWFObj.wfLength
    t_data = np.arange(dataLen) * 10
    ax0.plot(t_data, dataWFObj.windowedWf  ,color="red", lw=2, alpha=0.8)

    for simWFIdx in range(simwfnum):
      simWF = simWFArray[simWFIdx, dataIdx, :dataLen]
      diff = simWF - dataWFObj.windowedWf

      colorIdx = dataIdx
      if dataIdx >= len(colors): colorIdx =  dataIdx % len(colors)
      ax0.plot(t_data, simWF  , color=colors[colorIdx], alpha = residAlpha  )
      ax1.plot(t_data, diff  , color=colors[colorIdx],  alpha = residAlpha )

  legend_line_1 = ax0.plot( np.NaN, np.NaN, color='r', label='Data (unfiltered)' )
  legend_line_2 = ax0.plot( np.NaN, np.NaN, color='black', label='Fit waveform' )

  first_legend = ax0.legend(loc=4)
  ax1.set_ylim(-20,20)
#  ax0.set_ylim(6200,6500)

########################################################################
def plotWaveformParams(wfs, stepsFig, sampler, numWaveforms, nwalkers ):
  plt.figure(stepsFig.number)
  plt.clf()
  ax0 = stepsFig.add_subplot(611)
  ax1 = stepsFig.add_subplot(612, sharex=ax0)
  ax2 = stepsFig.add_subplot(613, sharex=ax0)
  ax3 = stepsFig.add_subplot(614, sharex=ax0)
  ax4 = stepsFig.add_subplot(615, sharex=ax0)
  ax5 = stepsFig.add_subplot(616, sharex=ax0)

  ax0.set_ylabel('r')
  ax1.set_ylabel('phi')
  ax2.set_ylabel('z')
  ax3.set_ylabel('scale')
  ax4.set_ylabel('t0')
  ax5.set_ylabel('smoothing')

  colors = ["red", "blue" , "green", "orange", "purple", "black", "cyan"]

  for i in range(nwalkers):
    for j in range(wfs.size):
      ax0.plot(sampler.chain[i,:,0+j], alpha=0.3, color="r")                 # r
      ax1.plot(sampler.chain[i,:,numWaveforms + j], alpha=0.3,  color=colors[j])    # phi
      ax2.plot(sampler.chain[i,:,2*numWaveforms + j], alpha=0.3,  color=colors[j])  #z
      ax3.plot(sampler.chain[i,:,3*numWaveforms + j],  alpha=0.3,  color=colors[j]) #energy
      ax4.plot(sampler.chain[i,:,4*numWaveforms + j],  alpha=0.3,  color=colors[j]) #t0
      ax5.plot(sampler.chain[i,:,5*numWaveforms + j],  alpha=0.3,  color=colors[j]) #smoothing


def findTimePoint(data, percent, timePointIdx=0):

  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)

#  print "finding percent %0.4f" % percent
#  print np.where(np.greater(int_data, percent))[0]
#  print np.where(np.greater(int_data, percent))[0][timePointIdx]

  if timePointIdx == 0:
    #this will only work assuming we don't hit
#    maxidx = np.argmax(int_data)
    return np.where(np.less(int_data, percent))[0][-1]

  if timePointIdx == -1:
    return np.where(np.greater(int_data, percent))[0][timePointIdx]

  else:
    print "timepointidx %d is not supported" % timePointIdx
    exit(0)
