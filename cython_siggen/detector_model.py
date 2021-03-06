#!/usr/local/bin/python

#import sys
import numpy as np
import copy, math
from scipy import  signal, interpolate, ndimage

from siggen import Siggen

#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file, temperature=0, timeStep=None, numSteps=None,):

    self.conf_file = siggen_config_file

    if timeStep is None or numSteps is None:
      self.siggenInst = Siggen(siggen_config_file)
    else:
      self.siggenInst =  Siggen(siggen_config_file, timeStep, numSteps)
      self.num_steps = np.int(numSteps)
      self.time_step_size = timeStep

      print "Time step size is %d" % self.time_step_size
      print "There will be %d steps in output" % self.num_steps

    (self.detector_radius, self.detector_length) = self.siggenInst.GetDimensions()

    print "Using model-based velocity numbers..."
    self.siggenInst.set_velocity_type(1)

    if temperature > 0:
      self.SetTemperature(temperature)

    #stuff for field interp
    self.wp_function = None
    self.efld_r_function = None
    self.efld_z_function = None
    self.rr = None
    self.zz = None
    self.wp_pp = None

    self.trapping_rc = None

    #stuff for waveform interpolation
    #round here to fix floating point accuracy problem
    data_to_siggen_size_ratio = np.around(10. / self.time_step_size,3)
    if not data_to_siggen_size_ratio.is_integer():
      print "Error: siggen step size must evenly divide into 10 ns digitization period (ratio is %f)" % data_to_siggen_size_ratio
      exit(0)
    elif data_to_siggen_size_ratio < 10:
      round_places = 0
    elif data_to_siggen_size_ratio < 100:
      round_places = 1
    elif data_to_siggen_size_ratio < 1000:
      round_places = 2
    else:
      print "Error: Ben was too lazy to code in support for resolution this high"
      exit(0)
    self.data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)

    #Holders for wf simulation
    self.calc_length = self.siggenInst.GetCalculationLength()

    self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
    self.raw_charge_data = np.zeros( self.calc_length, dtype=np.dtype('f4'), order="C" )
    self.processed_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )

###########################################################################################################################
  def LoadFields(self, fieldFileName):
    self.fieldFileName = fieldFileName

    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']
      pcRadList = data['pcRadList']
      pcLenList = data['pcLenList']

    self.gradList = gradList
    self.pcRadList = pcRadList
    self.pcLenList = pcLenList

    r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
    z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

#    self.wp_functions = np.empty((wpArray.shape[0],wpArray.shape[1]), dtype=np.object)
#    self.efld_r_functions = np.empty((wpArray.shape[0],wpArray.shape[1]), dtype=np.object)
#    self.efld_z_functions = np.empty((wpArray.shape[0],wpArray.shape[1]), dtype=np.object)
    self.wpArray = wpArray
    self.efld_rArray = efld_rArray
    self.efld_zArray = efld_zArray
##
#    for r in range(wpArray.shape[0]):
#      for z in range(wpArray.shape[1]):
#        self.wp_functions[r,z] = interpolate.RectBivariateSpline(pcRadList, pcLenList, wpArray[r,z,:,:], kx=1, ky=1)
#        self.efld_r_functions[r,z] = interpolate.RegularGridInterpolator((gradList, pcRadList, pcLenList), efld_rArray[r,z,:,:,:])
#        self.efld_z_functions[r,z] = interpolate.RegularGridInterpolator((gradList, pcRadList, pcLenList), efld_zArray[r,z,:,:,:])
#
    self.wp_function = interpolate.RegularGridInterpolator((r_space, z_space, pcRadList, pcLenList), wpArray, )
    self.efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_rArray, )
    self.efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_zArray,)

    (self.rr, self.zz) = np.meshgrid(r_space, z_space)
###########################################################################################################################
  def LoadFieldsGrad(self, fieldFileName, pcLen, pcRad):
    self.fieldFileName = fieldFileName

    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']

    self.gradList = gradList
    self.pcLen = pcLen
    self.pcRad = pcRad

    self.siggenInst.ReadEFieldsFromArray(efld_rArray, efld_zArray, wpArray )

    r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
    z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

    self.wpArray = wpArray
    self.efld_rArray = efld_rArray
    self.efld_zArray = efld_zArray

    self.efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, ), efld_rArray, )
    self.efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList,), efld_zArray,)

    (self.rr, self.zz) = np.meshgrid(r_space, z_space)

  def SetFields(self, pcRad, pcLen, impurityGrad, method="full"):
    if method=="nearest":
#      print "WARNING: DOING A CHEAP FIELD SET"
      return self.SetFieldsByNearest(pcRad, pcLen, impurityGrad)
    else:
      return self.SetFieldsFullInterp(pcRad, pcLen, impurityGrad)

  def SetFieldsGradInterp(self, impurityGrad):

    self.impurityGrad = impurityGrad
    gradIdx = (np.abs(self.gradList-impurityGrad)).argmin()
    self.SetFieldsGradIdx(gradIdx)
    #     rr = self.rr
    #     zz = self.zz
    #     efld_r_function = self.efld_r_function
    #     efld_z_function = self.efld_z_function
    #
    #     gradgrad = np.ones_like(rr) * impurityGrad
    #
    #     points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), ], dtype=np.dtype('f4') ).T
    #
    #     new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    #     new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    #
    # #    grad_idx = find_nearest_idx(self.gradList, impurityGrad)
    # #    new_ef_r = np.copy(self.efld_rArray[:,:,grad_idx][:,:,0])
    # #    new_ef_z = np.copy(self.efld_zArray[:,:,grad_idx][:,:,0])
    #
    #     self.siggenInst.SetFields(new_ef_r, new_ef_z, self.wpArray)


  def SetFieldsGradIdx(self, gradIdx):
      self.siggenInst.SetActiveEfld(gradIdx)


  def SetFieldsFullInterp(self, pcRad, pcLen, impurityGrad):
    self.pcRad = pcRad
    self.pcLen = pcLen
    self.impurityGrad = impurityGrad

    rr = self.rr
    zz = self.zz
    wp_function = self.wp_function
    efld_r_function = self.efld_r_function
    efld_z_function = self.efld_z_function

    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad

    points_wp =  np.array([rr.flatten() , zz.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T

    new_wp = np.array(wp_function( points_wp ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")

    self.siggenInst.SetPointContact( pcRad, pcLen )
    self.siggenInst.SetFields(new_ef_r, new_ef_z, new_wp)

  def SetFieldsByNearest(self, pcRad, pcLen, impurityGrad):
    self.pcRad = pcRad
    self.pcLen = pcLen
    self.impurityGrad = impurityGrad

    grad_idx = find_nearest_idx(self.gradList, impurityGrad)
    rad_idx = find_nearest_idx(self.pcRadList, pcRad)
    len_idx = find_nearest_idx(self.pcLenList, pcLen)

    new_wp = np.copy(self.wpArray[:,:,rad_idx, len_idx][:,:,0])
    new_ef_r = np.copy(self.efld_rArray[:,:,grad_idx,rad_idx, len_idx][:,:,0])
    new_ef_z = np.copy(self.efld_zArray[:,:,grad_idx,rad_idx, len_idx][:,:,0])

    wp_function = self.wp_function
    efld_r_function = self.efld_r_function
    efld_z_function = self.efld_z_function

#    grad_idx = np.searchsorted(self.gradList, impurityGrad, side="left")
#    rad_idx = np.searchsorted(self.pcRadList, pcRad, side="left")
#    len_idx = np.searchsorted(self.pcLenList, pcLen, side="left")

##    wpArray = self.wpArray[:,:,rad_idx, len_idx]
##    wpArrayNext = self.wpArray[:,:,rad_idx-1, len_idx-1]
#    wpArray = self.efld_rArray[:,:,grad_idx, rad_idx, len_idx]
#    wpArrayNext = self.efld_rArray[:,:,grad_idx-1,rad_idx-1, len_idx-1]
#    wpArray[np.where(wpArray==0)] = np.nan
#    div= np.divide(np.subtract(wpArray, wpArrayNext), wpArray)
#    import matplotlib.pyplot as plt
#
#    div_true = np.zeros_like(div)
#    div_true[np.where(div > 0.01)] = 1
#
#    plt.imshow(div_true.T, origin='lower')
#    plt.colorbar()
#    plt.show()
#    exit(0)

    #do the interp for the closest... 5mm?
#    min_distance_r = 20
#    min_distance_z = 30
#    r_space = np.around(np.arange(min_distance_r, self.wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4')), 1)
#    z_space = np.around(np.arange(min_distance_z, self.wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4')),1)
#    rr, zz = np.meshgrid(r_space, z_space)
#    radrad = np.ones_like(rr) * pcRad
#    lenlen = np.ones_like(rr) * pcLen
#    gradgrad = np.ones_like(rr) * impurityGrad
#    points_wp =  np.array([rr.flatten() , zz.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
#    new_wp[min_distance_r*10:len(r_space)+min_distance_r*10,  min_distance_z*10:len(z_space)+min_distance_z*10] =  wp_function( points_wp ).reshape(rr.shape).T

    min_distance_r = 0
    min_distance_z = 0
    r_space = np.around(np.arange(min_distance_r, self.wpArray.shape[0]/10 , 0.1, dtype=np.dtype('f4')),1)
    z_space = np.around(np.arange(min_distance_z, self.wpArray.shape[1]/10 , 0.1, dtype=np.dtype('f4')),1)
    rr, zz = np.meshgrid(r_space, z_space)
    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    new_ef_r[min_distance_r*10:len(r_space)+min_distance_r*10,  min_distance_z*10:len(z_space)+min_distance_z*10] = efld_r_function( points_ef ).reshape(rr.shape).T

    min_distance_r = 0#1.5
    min_distance_z = 0#1.5
    r_space = np.around(np.arange(min_distance_r, self.wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4')),1)
    z_space = np.around(np.arange(min_distance_z, self.wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4')),1)
    rr, zz = np.meshgrid(r_space, z_space)
    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    new_ef_z[np.int(min_distance_r*10):len(r_space)+np.int(min_distance_r*10),  np.int(min_distance_z*10):len(z_space)+np.int(min_distance_z*10)] =  efld_z_function( points_ef ).reshape(rr.shape).T

#    import matplotlib.pyplot as plt
##    plt.imshow(np.abs(new_wp.T - self.wpArray[:,:,rad_idx, len_idx][:,:,0].T) , origin='lower'
#    plt.imshow(np.sqrt(np.add(new_ef_r.T**2, new_ef_z.T**2)) , origin='lower')
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(np.abs(new_ef_z.T - self.efld_zArray[:,:,grad_idx,rad_idx, len_idx][:,:,0].T)  , origin='lower')
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(np.abs(new_ef_r.T - self.efld_rArray[:,:,grad_idx,rad_idx, len_idx][:,:,0].T) , origin='lower')
#    plt.colorbar()
#
#    plt.show()
#    exit()

    self.siggenInst.SetPointContact( pcRad, pcLen )
    self.siggenInst.SetFields(new_ef_r, new_ef_z, new_wp)

###########################################################################################################################
  def ReinitializeDetector(self):
#    self.SetTemperature(self.temperature)
#    self.SetFields(self.pcRad, self.pcLen, self.impurityGrad)
    self.SetFieldsGradInterp( self.impurityGrad)
###########################################################################################################################
  def SetTemperature(self, h_temp, e_temp=0):
    self.temperature = h_temp

    if e_temp == 0:
      e_temp = h_temp

    self.siggenInst.SetTemperature(h_temp, e_temp)
###########################################################################################################################
  def SetTransferFunction(self, b_over_a, c, d, RC1_in_us, RC2_in_us, rc1_frac):
    self.num = [1., b_over_a, 0.]
    self.den = [1., 2.*c, d**2]

    RC1= 1E-6 * (RC1_in_us)
    self.rc1_for_tf = np.exp(-1./1E8/RC1)

    RC2 = 1E-6 * (RC2_in_us)
    self.rc2_for_tf = np.exp(-1./1E8/RC2)

    self.rc1_frac = rc1_frac


  def SetTransferFunctionByTF(self, num, den):
    #should already be discrete params
    (self.num, self.den) = (num, den)
###########################################################################################################################
  def IsInDetector(self, r, phi, z):
    taper_length = 4.5
    if r > np.floor(self.detector_radius*10.)/10. or z > np.floor(self.detector_length*10.)/10.:
      return 0
    elif r <0 or z <=0:
      return 0
    elif z <= self.pcLen and r <= self.pcRad:
      return 0
    elif z < taper_length and r > (self.detector_length - taper_length + z):
      return 0
    elif phi <0 or phi > np.pi/4:
      return 0
    elif r**2/self.pcRad**2 + z**2/self.pcLen**2 < 1:
      return 0
    else:
      return 1
###########################################################################################################################
  def GetSimWaveform(self, r,phi,z,scale, switchpoint,  numSamples, smoothing=None):
    sig_wf = self.GetRawSiggenWaveform(r, phi, z)
    if sig_wf is None:
      return None
    #smoothing for charge cloud size effects
    if smoothing is not None:
      ndimage.filters.gaussian_filter1d(sig_wf, smoothing, output=sig_wf)
    sim_wf = self.ProcessWaveform(sig_wf, numSamples, scale, switchpoint)
    return sim_wf
########################################################################################################
  def GetRawSiggenWaveform(self, r,phi,z, energy=1):

    x = r * np.sin(phi)
    y = r * np.cos(phi)
    self.raw_siggen_data.fill(0.)

    calcFlag = self.siggenInst.GetSignal(x, y, z, self.raw_siggen_data);
    if calcFlag == -1:
#      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    if np.amax(self.raw_siggen_data) == 0:
      print "found zero wf at r=%0.2f, phi=%0.2f, z=%0.2f (calcflag is %d)" % (r, phi, z, calcFlag)
      return None
    return self.raw_siggen_data

###########################################################################################################################
  def MakeSimWaveform(self, r,phi,z,scale, switchpoint,  numSamples, h_smoothing = None):

    self.raw_siggen_data.fill(0.)
    ratio = np.int(self.calc_length / self.num_steps)

#    print "ratio is %d" % ratio

    hole_wf = self.MakeRawSiggenWaveform(r, phi, z, 1)
    if hole_wf is None:
      return None

#    if h_smoothing is not None:
#      ndimage.filters.gaussian_filter1d(hole_wf, h_smoothing, output=hole_wf)

#    #currently just correct cloud size for holes
#    if cloud_size > 0:
#      self.siggenInst.ChargeCloudCorrect(hole_wf, cloud_size)

    self.raw_siggen_data += hole_wf[::ratio]

    #charge trapping (for holes only), currently not being used
    if self.trapping_rc is not None:
      trapping_rc = self.trapping_rc * 1E-6
      trapping_rc_exp = np.exp(-1./1E8/trapping_rc)
      self.raw_siggen_data= signal.lfilter([1., -1], [1., -trapping_rc_exp], self.raw_siggen_data)
      holes_collected_idx = np.argmax(self.raw_siggen_data)
      self.raw_siggen_data[holes_collected_idx:] = self.raw_siggen_data[holes_collected_idx]

    electron_wf = self.MakeRawSiggenWaveform(r, phi, z, -1)
    if electron_wf is  None:
      return None

#    if h_smoothing is not None:
#      ndimage.filters.gaussian_filter1d(electron_wf, h_smoothing, output=electron_wf)

    self.raw_siggen_data += electron_wf[::ratio]

    if h_smoothing is not None:
      ndimage.filters.gaussian_filter1d(self.raw_siggen_data, h_smoothing/ratio, output=self.raw_siggen_data)

    sim_wf = self.ProcessWaveform(self.raw_siggen_data, numSamples, scale, switchpoint)
    return sim_wf

  def MakeRawSiggenWaveform(self, r,phi,z, charge):
    #Has CALCULATION, not OUTPUT, step length (ie, usually 1ns instead of 10ns binning)
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    self.raw_charge_data.fill(0.)

    calcFlag = self.siggenInst.MakeSignal(x, y, z, self.raw_charge_data, charge);
    if calcFlag == -1:
      return None

    return self.raw_charge_data

########################################################################################################
  def ProcessWaveform(self, siggen_wf, outputLength, scale, switchpoint):
    '''Use interpolation instead of rounding'''
    siggen_len = self.num_steps #+ self.zeroPadding

    switchpoint_ceil = switchpoint



    #resample the siggen wf to the 10ns digitized data frequency w/ interpolaiton
    switchpoint_ceil= np.int( np.ceil(switchpoint) )
    samples_to_fill = (outputLength - switchpoint_ceil)
    siggen_interp_fn = interpolate.interp1d(np.arange(self.num_steps ), siggen_wf, kind="linear", copy="False", assume_sorted="True")
    siggen_start_idx = (switchpoint_ceil - switchpoint) * self.data_to_siggen_size_ratio

    sampled_idxs = np.arange(samples_to_fill)*self.data_to_siggen_size_ratio + siggen_start_idx

    self.processed_siggen_data.fill(0.)

    try:
      self.processed_siggen_data[switchpoint_ceil:outputLength] = siggen_interp_fn(sampled_idxs)
    except ValueError:
      print "Something goofy happened here during interp"
      print outputLength
      print siggen_wf.size
      print sampled_idxs[-10:]
      return None

    #filter for the damped oscillation
    self.processed_siggen_data[:outputLength]= signal.lfilter(self.num, self.den, self.processed_siggen_data[:outputLength])

    #filter for the exponential decay
    rc2_num_term = self.rc1_for_tf*self.rc1_frac - self.rc1_for_tf - self.rc2_for_tf*self.rc1_frac
    self.processed_siggen_data[:outputLength]= signal.lfilter([1., -1], [1., -self.rc1_for_tf], self.processed_siggen_data[:outputLength])
    self.processed_siggen_data[:outputLength]= signal.lfilter([1., rc2_num_term], [1., -self.rc2_for_tf], self.processed_siggen_data[:outputLength])


    smax = np.amax(self.processed_siggen_data[:outputLength])

    if smax == 0:
      return None

    self.processed_siggen_data[:outputLength] /= smax
    self.processed_siggen_data[:outputLength] *= scale

    return self.processed_siggen_data[:outputLength]
########################################################################################################
  #For pickling a detector object
  def __getstate__(self):
    # Copy the object's state from self.__dict__ which contains
    # all our instance attributes. Always use the dict.copy()
    # method to avoid modifying the original state.

    #manually do a deep copy of the velo data
    self.siggenSetup = self.siggenInst.GetSafeConfiguration()

    state = self.__dict__.copy()
    # Remove the unpicklable entries.
    del state['rr']
    del state['zz']
    del state['raw_siggen_data']
    del state['efld_r_function']
    del state['efld_z_function']
    del state['wp_function']
    del state['pcRadList']
    del state['gradList']
    del state['pcLenList']
    del state['siggenInst']

    return state

  def __setstate__(self, state):
    # Restore instance attributes
    self.__dict__.update(state)
    # Restore the previously opened file's state. To do so, we need to
    # reopen it and read from it until the line count is restored.

    self.siggenInst =  Siggen(savedConfig=self.siggenSetup)

    self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
    self.raw_charge_data = np.zeros( self.calc_length, dtype=np.dtype('f4'), order="C" )
    self.LoadFields(self.fieldFileName)
    self.siggenInst.set_velocity_type(1)
    self.siggenInst.ReadEFieldsFromArray(self.efld_rArray, self.efld_zArray, self.wpArray )

  def ReflectPoint(self, r,z):
    #algorithm shamelessly ripped from answer on http://stackoverflow.com/questions/3306838/algorithm-for-reflecting-a-point-across-a-line
    m = self.detector_length / self.detector_radius
    d = (r + m*z)/(1+m*m)
    new_r = 2*d-r
    new_z = 2*d*m - z

    return (new_r, new_z)

def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return [idx-1]
    else:
        return [idx]
#  def __del__(self):
#    del self.wp_pp
#    del self.siggenInst
