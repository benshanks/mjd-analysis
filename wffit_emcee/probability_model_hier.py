#!/usr/local/bin/python
import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import itertools
from multiprocessing import Pool
'''Fit detector properties given an array of waveforms'''


r_mult = 1.
z_mult = 1.
scale_mult = 100.

def initializeDetector(det):
  global detector
  detector = det
  detector.ReinitializeDetector()

def initializeDetectorAndWfs(det, wfs):
  global detector
  detector = det
  detector.ReinitializeDetector()

  global wf_arr
  wf_arr = wfs

#def init_wfs(wf_Arr):
#  global wf_arr
#  wf_arr = wf_Arr

def lnlike_detector(theta, *wfParams):
  '''assumes the data comes in w/ 10ns sampling period'''

  temp, impGrad, pcRad, pcLen = np.copy(theta[:4])
  tfStartIdx = 4
  num = [np.copy(theta[tfStartIdx]) *1E9 , np.copy(theta[tfStartIdx+1]) *1E17, np.copy(theta[tfStartIdx+2])*1E15 ]
  den = [1, np.copy(theta[tfStartIdx+3]) *1E7 , np.copy(theta[tfStartIdx+4]) *1E14, np.copy(theta[tfStartIdx+5])*1E18 ]
  
  temp *= 10.
  impGrad /= 100.
  print ">>>> temp: %0.2f, pcrad %0.6f, pclen %0.6f, impgrad = %0.4f" % (temp, pcRad, pcLen, impGrad)
  print ">>>>              num: " + str(num)
  print ">>>>              den: " + str(den)
  
  pool = wfParams[0]
  wfParams = np.array(wfParams[1:])
  r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr = wfParams[:].reshape((6, len(wf_arr)))

  gradList  = detector.gradList
  pcRadList =  detector.pcRadList
  pcLenList =  detector.pcLenList
  
  #Take care of detector business
  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
    return -np.inf
  if pcLen < pcLenList[0] or pcLen > pcLenList[-1]:
    return -np.inf
  if impGrad < gradList[0] or impGrad > gradList[-1]:
    return -np.inf
  if temp < 40 or temp > 120:
    return -np.inf
  
  totalLike = 0

  a_args = np.arange(r_arr.size)

  args = []

  for idx in a_args:
    args.append( [r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx], temp, pcRad, pcLen, impGrad, num, den, wf_arr[idx] ]  )

  results = pool.map(minimize_wf_star, args)

  for (idx, result) in enumerate(results):
  
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result["x"]
    wf_like = -1*result['fun']
    
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, wf_like/wf_arr[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth: %0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
    
    if not np.isfinite(wf_like):
      return -np.inf
    #NORMALIZE FOR WF LENGTH
    totalLike += wf_like / wf_arr[idx].wfLength
  
  print "  >>total likelihood: %0.3f" % totalLike

  return totalLike

def lnlike_detector_holdtf(theta, *wfParams):
  '''assumes the data comes in w/ 10ns sampling period'''

  temp, impGrad, pcRad, pcLen= np.copy(theta[:4])

  temp *= 10.
  impGrad /= 100.
  print ">>>> temp: %0.2f, impgrad = %0.4f, pcrad = %0.4f, pclen = %0.4f" % (temp, impGrad, pcRad, pcLen, )
  
  pool = wfParams[0]
  wfParams = np.array(wfParams[1:])
  
  r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr = wfParams[:].reshape((6, len(wf_arr)))

  gradList  = detector.gradList
  pcRadList =  detector.pcRadList
  pcLenList =  detector.pcLenList
  
  #Take care of detector business
  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
    return -np.inf
  if pcLen < pcLenList[0] or pcLen > pcLenList[-1]:
    return -np.inf
  if impGrad < gradList[0] or impGrad > gradList[-1]:
    return -np.inf
  if temp < 40 or temp > 120:
    return -np.inf
  
  totalLike = 0

  a_args = np.arange(r_arr.size)

  args = []

  for idx in a_args:
    args.append( [r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx], temp, pcRad, pcLen, impGrad,  wf_arr[idx] ]  )

  results = pool.map(minimize_wf_hold_tf_star, args)

  for (idx, result) in enumerate(results):
  
    r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx] = result["x"]
    wf_like = -1*result['fun']
    
    print "  >> wf %d (normalized likelihood %0.2f):" % (idx, wf_like/wf_arr[idx].wfLength)
    print "      r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth: %0.2f" % (r_arr[idx], phi_arr[idx], z_arr[idx], scale_arr[idx], t0_arr[idx], smooth_arr[idx])
    
    if not np.isfinite(wf_like):
      return -np.inf
    #NORMALIZE FOR WF LENGTH
    totalLike += wf_like / wf_arr[idx].wfLength
  
  print "  >>total likelihood: %0.3f" % totalLike

  return totalLike


def minimize_waveform_only(r, phi, z, scale, t0, smooth, wf,):

  bounds = [ (0,np.floor(detector.detector_radius*10.)/10. /r_mult ), (0, np.pi/4.), (0,np.floor(detector.detector_length*10.)/10./z_mult), (wf.wfMax/2/scale_mult, wf.wfMax*2/scale_mult), (0, 15), (0, 20)  ]

  constraints = [{ "type": 'ineq', "fun": constr_r  },
                 { "type": 'ineq', "fun": constr_theta  },
                 { "type": 'ineq', "fun": constr_z  },
                 { "type": 'ineq', "fun": constr_t0  },
                 { "type": 'ineq', "fun": constr_smooth  }]

  result = op.minimize(neg_lnlike_wf, [r, phi, z, scale, t0, smooth], args=(wf) ,method="Powell")
  
  #result = op.differential_evolution(neg_lnlike_wf, bounds, args=([wf]), polish=False )
  return result

def constr_r(x):
  if x[0] <0:
    return -1
  else:
    return np.floor(detector.detector_radius*10.)/10./r_mult  - x[0]

def constr_theta(x):
  if x[1] <0:
    return -1
  else:
    return np.pi/4 - x[1]
def constr_z(x):
  if x[2] <0:
    return -1
  else:
    return np.floor(detector.detector_length*10.)/10./z_mult - x[2]

def constr_smooth(x):
  #must be g.t. 0
  return x[5]
def constr_t0(x):
  #must be g.t. 0
  return x[4]

def minimize_waveform_only_star(a_b):
  return minimize_waveform_only(*a_b)

def lnlike_waveform(theta, wf):
  r, phi, z, scale, t0, smooth = np.copy(theta)
#  print "  r: %0.2f, phi: %0.3f, z: %0.2f, e: %0.2f, t0: %0.2f, smooth: %0.2f" % (r, phi, z, scale, t0, smooth)

  
  r *= r_mult
  z *= z_mult
  scale *= scale_mult
  
  if scale < 0 or t0 < 0 or smooth<0:
    return -np.inf

  if not detector.IsInDetector(r, phi, z):
    return -np.inf

  data = wf.windowedWf
  model_err = wf.baselineRMS

  model = detector.GetSimWaveform(r, phi, z, scale, t0, len(data), smoothing=smooth)
  
  if model is None:
    return -np.inf

  inv_sigma2 = 1.0/(model_err**2)

  return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def neg_lnlike_wf(theta, wf):
  return -1*lnlike_waveform(theta, wf)

def neg_lnlike_wf_star(*a_b):
  return neg_lnlike_wf(a_b)

def minimize_wf(r, phi, z, scale, t0, smooth, temp, pcRad, pcLen, impGrad, num, den, wf):

  if detector.temperature != temp:
    detector.SetTemperature(temp)
  if detector.pcLen != pcLen or detector.pcRad != pcRad or detector.impurityGrad != impGrad:
    detector.SetFields(pcRad, pcLen, impGrad)

  detector.SetTransferFunction(num, den)
  
  result = op.minimize(neg_lnlike_wf, [r, phi, z, scale, t0, smooth], args=(wf) ,method="Powell")

  return result

def minimize_wf_star(a_b):
  return minimize_wf(*a_b)

def minimize_wf_holdtf(r, phi, z, scale, t0, smooth, temp, pcRad, pcLen, impGrad, wf):

  if detector.temperature != temp:
    detector.SetTemperature(temp)
  if detector.pcLen != pcLen or detector.pcRad != pcRad or detector.impurityGrad != impGrad:
    detector.SetFields(pcRad, pcLen, impGrad)
  
  result = op.minimize(neg_lnlike_wf, [r, phi, z, scale, t0, smooth], args=(wf) ,method="Powell")

  return result

def minimize_wf_hold_tf_star(a_b):
  return minimize_wf_holdtf(*a_b)

def IsAllowableStep(f_new, x_new, f_old, x_old):
  temp, impGrad, pcRad, pcLen = (x_new[:4])

  gradList  = detector.gradList
  pcRadList =  detector.pcRadList
  pcLenList =  detector.pcLenList
  
  #Take care of detector business
  if pcRad < pcRadList[0] or pcRad > pcRadList[-1]:
    return False
  if pcLen < pcLenList[0] or pcLen > pcLenList[-1]:
    return False
  if impGrad < gradList[0] or impGrad > gradList[-1]:
    return False
  if temp < 40 or temp > 120:
    return False

  return True




