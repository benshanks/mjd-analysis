#!/usr/local/bin/python
from ROOT import *

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"

detectorName = "P3KJR"

baselineSamples = 800

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
  

  def WindowWaveform(self, numSamples, earlySamples=20, t0riseTime = 0.005):
    '''Windows to a given number of samples'''
    self.wfMax = np.amax(self.waveformData)

    startGuess = self.EstimateT0()
    firstFitSampleIdx = startGuess-earlySamples
    lastFitSampleIdx = firstFitSampleIdx + numSamples
    
    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = earlySamples
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def WindowWaveformTimepoint(self, earlySamples=20, fallPercentage=None):
    '''Does "smart" windowing by guessing t0 and wf max'''
    self.wfMax = np.amax(self.waveformData)

    startGuess = self.EstimateT0()
    firstFitSampleIdx = startGuess-earlySamples
    
    lastFitSampleIdx = self.EstimateFromMax(fallPercentage)
    
    np_data_early = self.waveformData[firstFitSampleIdx:lastFitSampleIdx]

    self.t0Guess = earlySamples
    self.windowedWf = np_data_early
    self.wfLength = len(np_data_early)

  def EstimateT0(self):
    return np.where(np.less(self.waveformData, self.baselineRMS))[0][-1]

  def EstimateFromMax(self, fallPercentage=None):
  
    if fallPercentage is None:
      searchValue =  self.wfMax - self.baselineRMS
    else:
      searchValue = fallPercentage * self.wfMax
    return np.where(np.greater(self.waveformData, searchValue))[0][-1]


def GetWaveformByEntry(runNumber, entryNumber, channelNumber, doBaselineSub=True):
    gatFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s%d.root" % (dataSetName, detectorName, gatDataName, runNumber  ) )
    builtFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/built/%s/%s%d.root" % (dataSetName, detectorName, builtDataName, runNumber  ) )

    gat_file = TFile.Open(gatFilePath)
    gatTree = gat_file.Get(gatTreeName)
    built_file = TFile.Open(builtFilePath)
    builtTree = built_file.Get(builtTreeName)
    builtTree.AddFriend(gatTree)

    waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)
    waveform.SetLength(waveform.GetLength()-10)
    
    rms=0
    if doBaselineSub:
      baseline = MGWFBaselineRemover()
      baseline.SetBaselineSamples(baselineSamples)
      baseline.TransformInPlace(waveform)
      rms = baseline.GetBaselineRMS()

    np_data = np.array(waveform.GetVectorData())

    return Waveform(np_data, channelNumber, runNumber, entryNumber ,rms)


########################################################################

def GetWaveforms(runRanges, channelNumber, numWaveforms, cutStr):
  '''cut string should include all cuts except for channel cut'''

  cutChan = "channel == %d" % channelNumber
  cutStr = cutChan + " && " + cutStr

  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(baselineSamples)
  
  waveformArray = []
  
  runList = []
  for runs in runRanges:
    for run in range(runs[0], runs[1]+1):
      runList.append(run)

  for iRun in runList:
    print 'processing run', iRun
    gatFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s%d.root" % (dataSetName, detectorName, gatDataName, iRun  ) )
    builtFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/built/%s/%s%d.root" % (dataSetName, detectorName, builtDataName, iRun  ) )
    
    if not os.path.isfile(gatFilePath):
      print ">>>Skipping file " + gatFilePath
      continue

    gat_file = TFile.Open(gatFilePath)
    gatTree = gat_file.Get(gatTreeName)
    built_file = TFile.Open(builtFilePath)
    builtTree = built_file.Get(builtTreeName)
    builtTree.AddFriend(gatTree)
    
    gatTree.SetEntryList(0)
    gatTree.Draw(">>elist", cutStr, "entrylist")
    elist = gDirectory.Get("elist")
    #print "Number of entries in the entryList is " + str(elist.GetN())

    
    gatTree.Draw(">>elistChan", cutChan, "entrylist")
    elistChan = gDirectory.Get("elistChan")

    gatTree.SetEntryList(elist);
    builtTree.SetEntryList(elist);
    
    for ientry in xrange( elist.GetN() ):
      entryNumber = gatTree.GetEntryNumber(ientry);
      
#      if entryNumber in funnyEntries: continue

      waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)
      waveform.SetLength(waveform.GetLength()-10)
      baseline.TransformInPlace(waveform)
      
      print "Waveform number %d in run %d" % (entryNumber, iRun)
      #fitWaveform(waveform, fig, fig2, iRun, entryNumber, channelNumber)
      
      np_data = waveform.GetVectorData()
      np_data = np.array(np_data)
      
#      event = builtTree.event
#      current_time = event.GetTime()
#      
#      #find the last hit event for this channel
#      for i in range(elistChan.GetN()):
#        e_num_tmp = elistChan.GetEntry(i)
#        if e_num_tmp >= entryNumber: break
#        last_entry_number = e_num_tmp
#      
#      builtTree.GetEntry(last_entry_number)
#      lastEvent = builtTree.event
#      
#      timeSinceLast =  current_time - lastEvent.GetTime()
#
#      energyLast = getWaveformEnergy(gatTree, builtTree, last_entry_number, channelNumber)

      energy = getWaveformEnergy(gatTree, builtTree, entryNumber, channelNumber)

      waveformArray.append( Waveform(np_data, channelNumber, iRun, entryNumber ,baseline.GetBaselineRMS(),  energy=energy) )
      if len(waveformArray) >= numWaveforms: break
      
    gat_file.Close()
    built_file.Close()
    if len(waveformArray) >= numWaveforms: break

  waveformArrayNumpy = np.array(waveformArray)

  return waveformArrayNumpy


########################################################################

def getWaveform(gatTree, builtTree, entryNumber, channelNumber):
    
    builtTree.GetEntry( entryNumber )
    gatTree.GetEntry( entryNumber )
    
    event = builtTree.event
    channelVec   = gatTree.channel
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        return event.GetWaveform(i_wfm)

def getWaveformEnergy(gatTree, builtTree, entryNumber, channelNumber):
    
    builtTree.GetEntry( entryNumber )
    gatTree.GetEntry( entryNumber )
    
    event = builtTree.event
    channelVec   = gatTree.channel
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        return gatTree.trapENFCal[i_wfm]


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

  for (dataIdx, dataWFObj) in enumerate(dataWFArray):
    
    dataLen = dataWFObj.wfLength
    t_data = np.arange(dataLen) * 10
    ax0.plot(t_data, dataWFObj.windowedWf  ,color="red", lw=2, alpha=0.8)

    for simWFIdx in range(simwfnum):
      simWF = simWFArray[simWFIdx, dataIdx, :dataLen]
      diff = simWF - dataWFObj.windowedWf

      ax0.plot(t_data, simWF  ,color="black", alpha = residAlpha  )
      ax1.plot(t_data, diff  ,color="#7BAFD4",  alpha = residAlpha )

  legend_line_1 = ax0.plot( np.NaN, np.NaN, color='r', label='Data (unfiltered)' )
  legend_line_2 = ax0.plot( np.NaN, np.NaN, color='black', label='Fit waveform' )

  first_legend = ax0.legend(loc=4)

########################################################################


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


##################################
# From Clara's /global/project/projectdirs/majorana/data/production/P3KJR_AoverE_160602_85.dat on pdsf
channelAEDict = {
582: "(abs(TSCurrent200nsMax/trapENF)-(-0.00000000819708*(trapENFCal)))/0.00464859",
580:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000002478864*(trapENFCal)))/0.00459782",
578:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000002231200*(trapENFCal)))/0.00705478",
592:"(abs(TSCurrent200nsMax/trapENF)-(-0.00000001718770*(trapENFCal)))/0.00496213",
594:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000000195205*(trapENFCal)))/0.00679792",
598:"(abs(TSCurrent100nsMax/trapENF)-(-0.00000001621977*(trapENFCal)))/0.00622601",
600:"(abs(TSCurrent100nsMax/trapENF)-(-0.00000002458613*(trapENFCal)))/0.00601021",
608:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000001808628*(trapENFCal)))/0.00731791",
610:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000002479243*(trapENFCal)))/0.00710022",
616:"(abs(TSCurrent200nsMax/trapENF)-(-0.00000010087226*(trapENFCal)))/0.00451685",
626:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000001891262*(trapENFCal)))/0.00679347",
632:"(abs(TSCurrent100nsMax/trapENF)-(-0.00000001715065*(trapENFCal)))/0.00622425",
640:"(abs(TSCurrent100nsMax/trapENF)-(-0.00000001044140*(trapENFCal)))/0.00661752",
648:"(abs(TSCurrent200nsMax/trapENF)-(-0.00000001321028*(trapENFCal)))/0.00492125",
664:"(abs(TSCurrent100nsMax/trapENF)-(-0.00000006104607*(trapENFCal)))/0.00554796",
672:"(abs(TSCurrent200nsMax/trapENF)-(-0.00000000928838*(trapENFCal)))/0.00496773",
690:"(abs(TSCurrent50nsMax/trapENF)-(-0.00000002878251*(trapENFCal)))/0.00689086",
692:"(abs(TSCurrent100nsMax/trapENF)-(-0.00000001644495*(trapENFCal)))/0.00623492"
}


def getChannelAECut(channel):
  return channelAEDict[channel]
def getHighGainChannels():
  return channelAEDict.keys()


