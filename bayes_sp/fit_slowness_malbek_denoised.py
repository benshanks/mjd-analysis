#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
#ROOT.gApplication.ExecuteFile("$DISSDIR/Data/figure_style.C")

import matplotlib
matplotlib.use('pdf') #use a non-interactive backend that can handle multiprocessing
import matplotlib.pyplot as plt

import sys, os, csv, array, pymc
import numpy as np
from scipy import ndimage
import slowpulse_model_mc2 as sm

import pywt, math
import scipy as sp

from guppy import hpy
from multiprocessing import Process, Manager

#Graham's directory prefix
#dirPrefix = '$DISSDIR/Data'

#Ben's direcoty prefix
dirPrefix = '$MJDDATADIR/malbek/'

doPlots=1
writeFile = 1
interactive = 0 #run wf by wf


newTreeName = "spParamSkim_190_wavelet.root"
outputDir = "output190_wavelet"


flatTimeSamples = 2000 #in number of samples, not time
wl = pywt.Wavelet('haar')
levels = 8


def main(argv):

  setupDirs(outputDir)

  if interactive:
    plt.ion()
    fig = plt.figure(1) #waveform fit plot
    fig2 = plt.figure(2)

  #Instantiate and prepare the baseline remover transform:
  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(flatTimeSamples)

  #Load the malbek t4 tree without any RT cuts
  file = TFile(dirPrefix + 'ThesisAll_newCal.root')
  tree_nort = file.Get('fTree_noRT')

  #Load the malbek t4 waveform tree
  wf_file = TFile(dirPrefix + 't4_wf_all.root')
  tree_wf = wf_file.Get('tree')

  #Build an index for wf tree for later access
  tree_wf.BuildIndex('run_id', 'event_id')
  
  tree_nort.AddFriend(tree_wf)

  #Loop through events within a given energy range
  energy_low = 0.6
  energy_high = 10.
  count = 0
  
  if writeFile:
    oFile = TFile(newTreeName,"recreate");
    oFile.cd();
    outTree = TTree("slowpulseParamTree","Slowpulse Param");
    energy = np.zeros(1, dtype=float)
    spParam = np.zeros(1, dtype=float)
    riseTime = np.zeros(1, dtype=float)
    wpar = np.zeros(1, dtype=float)
    outTree.Branch("energykeV",energy, "energykeV/D");
    outTree.Branch("spParam",spParam, "spParam/D");
    outTree.Branch("riseTime",riseTime, "riseTime/D");
    outTree.Branch("wpar",wpar, "wpar/D");

  energyCut = "rfEnergy_keV>%f && rfEnergy_keV<%f" % (energy_low,energy_high)


  cut = energyCut
#  riseTimeCut = "rfRiseTime>2000"
#
#  cut = "%s && %s" % (energyCut, riseTimeCut)

  tree_nort.SetEntryList(0)
  tree_nort.Draw(">>elist", cut, "entrylist")
  elist = gDirectory.Get("elist")
  tree_nort.SetEntryList(elist);
  tree_wf.SetEntryList(elist);
  numEntries = elist.GetN()
  print "Total number of entries (w/ energy cut): %d" % numEntries

#numEntries = tree_nort.GetEntries()

  manager = Manager()
  return_dict = manager.dict()

  for i in xrange( numEntries):
    print "Entry %d of %d" % (i, numEntries)
    entryNumber = tree_nort.GetEntryNumber(i);
    #    entryNumber = i

    tree_nort.GetEntry(entryNumber)
    tree_wf.GetEntry(entryNumber)
  
    #Check to see if wf is in the energy range  
    #if tree_nort.rfEnergy_keV > energy_low and tree_nort.rfEnergy_keV < energy_high:
    current_run = tree_nort.rfRunID
    current_id = tree_nort.rfEventID   
    if tree_wf.GetEntryWithIndex(current_run, current_id) == -1:
      print 'waveform %s, %s not found' % (current_run, current_id)
      break
    waveform = tree_wf.waveform.Clone()
          
    #Baseline subtract it here (should eventually be incorporated into the model)
    baseline.TransformInPlace(waveform)
    
    #MCMC fit and plot the results
    p = Process(target=fitWaveform, args=(waveform, tree_nort.rfEnergy_keV, entryNumber, return_dict))
    #spParamTemp = fitWaveform(waveform, tree_nort.rfEnergy_keV)
    p.start()
    p.join()

    spParamTemp = return_dict["spParam"]

    if interactive:
      print "risetime:     %f" % tree_nort.rfRiseTime
    if writeFile:
      energy[0] = tree_nort.rfEnergy_keV
      riseTime[0] = tree_nort.rfRiseTime
      spParam[0] = spParamTemp
      wpar[0] = tree_nort.rfWpar
      outTree.Fill();

  if writeFile:
    oFile.Write();
    oFile.Close()

def fitWaveform(wf, energy, entryNumber, returnDict):

  np_data_unfiltered = wf.GetVectorData()
  threshold_list = get_threshold_list()
  
  swt_output = pywt.swt(np_data_unfiltered, wl, level=levels)
  # threshold the SWT coefficients
  apply_threshold(swt_output, 1., threshold_list)
  # inverse transform
  cA_thresh = iswt(swt_output, wl)
  
  np_data = cA_thresh

  #re-baseline-subtract
  np_data -= np.mean(np_data[:flatTimeSamples])


  wfMax = np.amax(np_data)
  
  lastFitSampleIdx = 4300
  fitSamples = 2000 #can't be longer than 800 right now (that's the length of the siggen wf...)

  firstFitSampleIdx = lastFitSampleIdx - fitSamples
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
  
  startGuess_9kev = 3850
  startGuess_p6kev = 3450
  
  startGuess = startGuess_p6kev + (startGuess_9kev - startGuess_p6kev) * (energy -0.6)/9
  
  t0_guess = startGuess - firstFitSampleIdx
  startVal = startGuess
  baseline_guess = 0
  energy_guess = 1000 * energy / 5 #normalized for 5 keV i guess.  i dunno.  whatever.
  noise_sigma_guess = np.std(np_data[0:flatTimeSamples])
  
  iterations = 2000
  burnin = iterations-500
  #adaptiveDelay = 100
  
  
  #in case you gotta plot wtf is going on before the fit
#  plt.figure(1)
#  plt.clf()
#  #plt.title("Charge waveform")
#  plt.xlabel("Digitizer samples")
#  plt.ylabel("Raw ADC Value [Arb]")
#  plt.plot(np_data  ,color="red" )
#  value = raw_input('  --> Press q to quit, any other key to continue\n')
#  if value == 'q':
#    exit(1)

  if doPlots:
    verbosity = 1
  else:
    verbosity = 0

  
  siggen_model = pymc.Model( sm.createSignalModelSiggen(np_data_early, t0_guess, energy_guess, noise_sigma_guess, baseline_guess) )
  M = pymc.MCMC(siggen_model, verbose=0)#, db="txt", dbname="Event_%d" % entryNumber)
  M.use_step_method(pymc.Metropolis, M.slowness_sigma, proposal_sd=1., proposal_distribution='Normal')
  M.use_step_method(pymc.Metropolis, M.wfScale, proposal_sd=10., proposal_distribution='Normal')
  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_sd=1., proposal_distribution='Normal')

#M.use_step_method(pymc.AdaptiveMetropolis, [M.slowness_sigma, M.wfScale, M.switchpoint], , shrink_if_necessary=1)
#  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale], delay=1000)
#  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_distribution='Normal', proposal_sd=4)
  M.sample(iter=iterations, verbose=0)

  t0 = np.around( np.median(M.trace('switchpoint')[burnin:]))
  scale =  np.median(M.trace('wfScale')[burnin:])
  sigma =  np.median(M.trace('slowness_sigma')[burnin:])
#  baselineB =  np.median(M.trace('baselineB')[burnin:])
#  baselineM =  0#np.median(M.trace('baselineM')[burnin:])
  startVal = t0 + firstFitSampleIdx

  returnDict["spParam"] = sigma
#  print ">>> noise_sigma:    %f" % (M.trace('noise_sigma')[-1])**(-.5)

  M.halt()

  if doPlots:
    print ">>> startVal:    %d" % startVal
    print ">>> scale:       %f" % scale
    print ">>> slowness:    %f" % sigma
  
#########  Plots for MC Steps
    stepsFig = plt.figure()
    plt.clf()
    ax0 = stepsFig.add_subplot(311)
    ax1 = stepsFig.add_subplot(312, sharex=ax0)
    ax2 = stepsFig.add_subplot(313, sharex=ax0)
    
    ax0.plot(M.trace('switchpoint')[:])
    ax0.set_ylabel('t0')
    ax1.plot(M.trace('slowness_sigma')[:])
    ax1.set_ylabel('slowness')
    ax2.plot(M.trace('wfScale')[:])
    ax2.set_ylabel('energy')
    
  #  axarr[3].plot(M.trace('noise_sigma')[:])
  #  axarr[3].set_ylabel('noise_sigma')

    ax2.set_xlabel('MCMC Step Number')
    ax0.set_title('Raw MCMC Sampling')

    stepsFig.savefig(outputDir + "/energy%d/mcsteps_Event%d_energy%0.3f_spparam%0.3f.pdf" % (floor(energy), entryNumber, energy, sigma))

  #########  Waveform fit plot

    detZ = np.floor(30.0)/2.
    detRad = np.floor(30.3)
    phiAvg = np.pi/8
    siggen_fit = sm.findSiggenWaveform(detRad, phiAvg, detZ)
    siggen_fit *= scale

    out = np.zeros(len(np_data_early))
    out[t0:] += siggen_fit[0:(len(siggen_fit) - t0)]
    out = ndimage.filters.gaussian_filter1d(out, sigma)

    f = plt.figure()
    plt.clf()
    #plt.title("Charge waveform")
    plt.xlabel("Digitizer time [ns]")
    plt.ylabel("Raw ADC Value [Arb]")
    plt.plot(np.arange(0, len(np_data)*10, 10), np_data  ,color="red" )
    plt.xlim( firstFitSampleIdx*10, (lastFitSampleIdx+25)*10)

    plt.plot(np.arange(firstFitSampleIdx*10, lastFitSampleIdx*10, 10), out  ,color="blue" )
  #  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
  #  plt.xlim( startVal-10, startVal+10)
  #  plt.ylim(-10, 25)
    plt.axvline(x=lastFitSampleIdx*10, linewidth=1, color='r',linestyle=":")
    plt.axvline(x=startVal*10, linewidth=1, color='g',linestyle=":")

    f.savefig(outputDir + "/energy%d/wf_Event%d_energy%0.3f_spparam%0.3f.pdf" % (floor(energy), entryNumber, energy, sigma))


#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    if value == 'q':
#      exit(1)

def setupDirs(outputDir):
  os.makedirs(outputDir)
  for i in xrange(0,10):
    os.makedirs(outputDir + "/energy%d" % i)

def get_threshold_list():
 return [ 206.2,
          575.1,
          447.7,
          313.6,
          229.1,
          207.2,
          524.8,
          780.2]

def iswt(coefficients, wavelet):
    """
      Input parameters: 

        coefficients
          approx and detail coefficients, arranged in level value 
          exactly as output from swt:
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

    """
    output = coefficients[0][0].copy() # Avoid modification of input data

    #num_levels, equivalent to the decomposition level, n
    num_levels = len(coefficients)
    for j in range(num_levels,0,-1): 
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        _, cD = coefficients[num_levels - j]
        for first in range(last_index): # 0 to last_index - 1

            # Getting the indices that we will transform 
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2] 
            # select the odd indices
            odd_indices = indices[1::2] 

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per') 
            x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per') 

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.  

    return output
    
def apply_threshold(output, scaler = 1., input=None):
   """ 
       output is a list of vectors (cA and cD, approximation
       and detail coefficients) exactly as you would expect
       from swt decomposition.  
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

       If input is none, this function will calculate the
       tresholds automatically for each waveform.
       Otherwise it will use the tresholds passed in, assuming
       that the length of the input is the same as the length
       of the output list.
       input looks like:
          [threshold1, threshold2, ..., thresholdn]

       scaler is a tuning parameter that will be multiplied on
       all thresholds.  Default = 1 (0.8?)
   """
      
   for j in range(len(output)):
      cA, cD = output[j]
      if input is None:
        dev = np.median(np.abs(cD - np.median(cD)))/0.6745
        thresh = math.sqrt(2*math.log(len(cD)))*dev*scaler
      else: thresh = scaler*input[j]
      cD = pywt.threshold(cD, thresh, mode='hard')
      output[j] = (cA, cD)

def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]

if __name__=="__main__":
    main(sys.argv[1:])


