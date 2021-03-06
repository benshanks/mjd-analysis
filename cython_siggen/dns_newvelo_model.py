import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng
from multiprocessing import Pool
from pysiggen import Detector

def initializeDetector(det, reinit=True):
  global detector
  detector = det

def initializeWaveforms( wfs_init, ):
  global wfs
  wfs = wfs_init

  global num_waveforms
  num_waveforms = len(wfs)


def initializeDetectorAndWaveforms(det_state, wfs_init, reinit=True, doInterp=True):
  initializeWaveforms(wfs_init)
  initializeDetector(det_state, reinit)

  global doMaxInterp
  doMaxInterp = doInterp

def initMultiThreading(numThreads):
  global num_threads
  global pool
  num_threads = numThreads
  if num_threads > 1:
      pool = Pool(num_threads, )

def initT0Padding(maxt_pad, linear_baseline_origin):
    global maxt_guess, min_maxt, max_maxt, baseline_origin_idx
    maxt_guess = maxt_pad
    max_maxt = maxt_guess + 10
    min_maxt = maxt_guess - 10
    baseline_origin_idx = linear_baseline_origin

traprc_min = 150
tf_phi_max = -1.2

tf_first_idx = 0
velo_first_idx = 7

grad_idx = velo_first_idx + 6
imp_avg_idx = grad_idx + 1
trap_idx = imp_avg_idx + 1

ba_idx, c_idx, dc_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3
aliasrc_idx = tf_first_idx+6


ba_prior = 0.107213
c_prior = -0.808
dc_prior = 0.815/c_prior

# h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior = 66333., 0.744, 181.
# h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior =  107270., 0.580, 100.

E_a = 500
va_lims = [2.5E6, 10E6]
vmax_lims = [7.5E6, 1.5E7]
beta_lims = [0.4, 1]

priors = np.empty(trap_idx+1) #6 + 2)
prior_vars =  np.empty(len(priors))


prior_vars[rc1_idx:rc1_idx+3] = 0.2, 0.3, 0.001

priors[aliasrc_idx] = 3.
prior_vars[aliasrc_idx] = 3.

priors[grad_idx] = 100
prior_vars[grad_idx] = 3
priors[trap_idx] = 200.
prior_vars[trap_idx] = 25.

def get_velo_params():
    return (priors[velo_first_idx:velo_first_idx+6], velo_var)
def get_t0_params():
    return (t0_guess, min_t0, max_t0, )
def get_param_idxs():
    return (tf_first_idx, velo_first_idx,  grad_idx, trap_idx)

def draw_position(wf_idx):
  r = rng.rand() * detector.detector_radius
  z = rng.rand() * detector.detector_length
  scale = np.amax(wfs[wf_idx].windowedWf)
  t0 = None

  if not detector.IsInDetector(r, 0.1, z):
    return draw_position(wf_idx)
  else:
    return (r,z, scale, t0)


class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self, rc1, rc2, rcfrac):
        """
        Parameter values *are not* stored inside the class
        """
        self.changed_wfs = np.zeros(num_waveforms)
        self.ln_likes = np.zeros(num_waveforms)


        rc1_prior =  rc1
        rc2_prior = rc2
        rc_frac_prior = rcfrac

        priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior

        h100_v500_meas = 7.156476E6
        h111_v500_meas = 6.056016E6
        h100_mu0E0_meas = 12.006273E6
        h111_mu0E0_meas = 10.727000E6

        priors[velo_first_idx] = h100_v500_meas
        priors[velo_first_idx+1] = h111_v500_meas
        priors[velo_first_idx+2] = h100_mu0E0_meas
        priors[velo_first_idx+3] = h111_mu0E0_meas

        for i in range(4):
            prior_vars[velo_first_idx+i] = 0.2*priors[velo_first_idx+i]

        priors[grad_idx] = detector.measured_imp_grad
        priors[grad_idx+1] = detector.measured_impurity

        print ("grad: %f" % detector.measured_imp_grad)
        print ("imp: %f" % detector.measured_impurity)

        prior_vars[grad_idx] =  np.amax((detector.measured_imp_grad, 0.02))
        prior_vars[grad_idx+1] = np.abs(0.2*detector.measured_impurity)

    def from_prior(self):
        """
        Unlike in C++, this must *return* a numpy array of parameters.
        """
        r_arr      = np.empty(num_waveforms)
        z_arr      = np.empty(num_waveforms)

        rad_arr      = np.empty(num_waveforms)
        phi_arr    = np.empty(num_waveforms)
        theta_arr  = np.empty(num_waveforms)
        scale_arr  = np.empty(num_waveforms)
        t0_arr     = np.empty(num_waveforms)
        smooth_arr = np.empty(num_waveforms)
        # m_arr      = np.empty(num_waveforms)
        # b_arr      = np.empty(num_waveforms)

        #draw 8 waveform params for each waveform
        for (wf_idx, wf) in enumerate(wfs):
            (r,z, scale, t0) = draw_position(wf_idx)
            smooth_guess = 10
            rad = np.sqrt(r**2+z**2)
            theta = np.arctan(z/r)

            r_arr[wf_idx] = r
            z_arr[wf_idx] = z
            rad_arr[wf_idx] = rad
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            theta_arr[wf_idx] = theta
            scale_arr[wf_idx] = 5*rng.randn() + scale - .005*scale
            t0_arr[wf_idx] = 3*rng.randn() + maxt_guess
            smooth_arr[wf_idx] = np.clip(rng.randn() + smooth_guess, 0, 20)
            # m_arr[wf_idx] =  0.0001*rng.randn() + 0.
            # b_arr[wf_idx] =  0.001*rng.randn() + 0.

        phi = (tf_phi_max + np.pi/2) * rng.rand() - np.pi/2
        omega = np.pi * rng.rand()
        d = rng.rand()

        #limit from 60 to 90
        rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 65, 80)
        rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0.1, 10)
        rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)

        aliasrc = dnest4.wrap(prior_vars[aliasrc_idx]*rng.randn() + priors[aliasrc_idx], 0.01, 10)

        # grad = rng.randint(len(detector.gradList))
        # avgImp = rng.randint(len(detector.impAvgList))
        charge_trapping = rng.rand()*(1000 - traprc_min) +  traprc_min

        grad = prior_vars[grad_idx]*rng.randn() + priors[grad_idx]
        avgImp = grad = prior_vars[grad_idx+1]*rng.randn() + priors[grad_idx+1]
        grad = dnest4.wrap(grad, detector.gradList[0], detector.gradList[-1])
        avgImp = dnest4.wrap(avgImp, detector.impAvgList[0], detector.impAvgList[-1])


        h_100_beta = (beta_lims[1] - beta_lims[0]) * rng.rand() + beta_lims[0]
        h_111_beta = (beta_lims[1] - beta_lims[0]) * rng.rand() + beta_lims[0]

        h_100_va = dnest4.wrap(prior_vars[velo_first_idx]*rng.randn() + priors[velo_first_idx], 1, 10*priors[velo_first_idx])
        h_111_va = dnest4.wrap(prior_vars[velo_first_idx+1]*rng.randn() + priors[velo_first_idx+1], 1, 10*priors[velo_first_idx])
        h_100_vmax = dnest4.wrap(prior_vars[velo_first_idx+2]*rng.randn() + priors[velo_first_idx+2], 1, 10*priors[velo_first_idx])
        h_111_vmax = dnest4.wrap(prior_vars[velo_first_idx+3]*rng.randn() + priors[velo_first_idx+3], 1, 10*priors[velo_first_idx])

        return np.hstack([
              phi, omega, d,
              #b, c, dc,
              rc1, rc2, rcfrac, aliasrc,
              h_100_va, h_111_va, h_100_vmax, h_111_vmax, h_100_beta, h_111_beta,
            #   k0_0, k0_1, k0_2, k0_3,
              grad, avgImp, charge_trapping,
              rad_arr[:], phi_arr[:], theta_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:],
            ])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0

        #decide whether to adjust just waveforms, or both wfs and detector params

        if rng.rand() <= 0.5:
            #adjust detector only
            reps = 1;
            if(rng.rand() < 0.5):
                reps += np.int(np.power(100.0, rng.rand()));

            for i in range(reps):
                which = rng.randint(len(priors))
                logH += self.perturb_detector(params, which)
        else:
            #adjust at least one waveform:
            self.changed_wfs.fill(0)
            randt2 = rng.randn()/np.sqrt(-np.log(rng.rand()));
            chance = np.power(10.0, -3*np.abs(randt2));

            for wf_idx in range(num_waveforms):
                if rng.rand() <= chance:
                     self.changed_wfs[wf_idx] = True
                #make sure one waveform is changed:
            if np.any(self.changed_wfs) == 0:
                self.changed_wfs[rng.randint(num_waveforms)] = 1

            for wf_idx in range(num_waveforms):
                if self.changed_wfs[wf_idx] == 1:
                    logH += self.perturb_wf(params, wf_idx)
        return logH

    def perturb_detector(self, params, which):
        logH = 0.0

        if which == ba_idx: #lets call this phi
           params[which] += (tf_phi_max + np.pi/2)*dnest4.randh()
           params[which] = dnest4.wrap(params[which], -np.pi/2, tf_phi_max)
        elif which == c_idx: #call it omega
            params[which] += 0.1*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.07, 0.2)
        elif which == dc_idx: #d
            params[which] += 0.01*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.7, 0.9)

        elif which == rc1_idx or which == rc2_idx or which == rcfrac_idx:
            #all normally distributed priors
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == aliasrc_idx:
          params[which] += 9.9*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 0.1, 10)

        elif which == grad_idx:
          logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
          params[which] += prior_vars[which] *dnest4.randh()
          params[which] = dnest4.wrap(params[which], detector.gradList[0], detector.gradList[-1])
          logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == imp_avg_idx:
          logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
          params[which] += prior_vars[which] *dnest4.randh()
          params[which] = dnest4.wrap(params[which], detector.impAvgList[0], detector.impAvgList[-1])
          logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == trap_idx:
            params[which] += (1000 - traprc_min)*dnest4.randh()
            params[which] = dnest4.wrap(params[which], traprc_min, 1000)

        elif which >= velo_first_idx and which < velo_first_idx+4:
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 1, priors[which]*10)
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == velo_first_idx+4 or which == velo_first_idx+5:
          params[which] += (beta_lims[1] - beta_lims[0])  *dnest4.randh()
          params[which] = dnest4.wrap(params[which], beta_lims[0], beta_lims[1])

        else: #velocity or rc params: cant be below 0, can be arb. large
            print ("which value %d not supported" % which)
            exit(0)

        return logH


    def perturb_wf(self, params, wf_idx, ):
    #do both wf and detector params in case theres strong correlation
        logH = 0.0

        reps = 1
        if rng.rand() < 0.5:
            reps += np.int(np.power(100.0, rng.rand()));

        for i in range(reps):
            wf_which = rng.randint(6)

            # my_which = rng.randint(len(priors) + 8)

            # if my_which < len(priors):
            #     #detector variable
            #     logH += self.perturb_detector(params, my_which)
            #
            # else:
            if wf_which < 6:
                #this is a waveform variable!
                # wf_which =  np.int(my_which - len(priors))

                #which idx of the global params array
                which = len(priors) + wf_which*num_waveforms + wf_idx

                rad_idx = len(priors) + wf_idx
                theta_idx =  len(priors) + 2*num_waveforms+ wf_idx
                self.changed_wfs[wf_idx] = 1

                if wf_which == 0:
                  theta = params[theta_idx]

                  #FIND THE MAXIMUM RADIUS STILL INSIDE THE DETECTOR
                  theta_eq = np.arctan(detector.detector_length/detector.detector_radius)
                  theta_taper = np.arctan(detector.taper_length/detector.detector_radius)
                #   print "theta: %f pi" % (theta / np.pi)
                  if theta <= theta_taper:
                     z = np.tan(theta)*(detector.detector_radius - detector.taper_length) / (1-np.tan(theta))
                     max_rad = z / np.sin(theta)
                  elif theta <= theta_eq:
                      max_rad = detector.detector_radius / np.cos(theta)
                    #   print "max rad radius: %f" %  max_rad
                  else:
                      theta_comp = np.pi/2 - theta
                      max_rad = detector.detector_length / np.cos(theta_comp)
                    #   print "max rad length: %f" %  max_rad

                  #AND THE MINIMUM (from PC dimple)
                  #min_rad  = 1./ ( np.cos(theta)**2/detector.pcRad**2  +  np.sin(theta)**2/detector.pcLen**2 )

                  min_rad = np.amax([detector.pcRad, detector.pcLen])

                  total_max_rad = np.sqrt(detector.detector_length**2 + detector.detector_radius**2 )

                  params[which] += total_max_rad*dnest4.randh()
                  params[which] = dnest4.wrap(params[which] , min_rad, max_rad)

                elif wf_which ==2: #theta
                  rad = params[rad_idx]

                #   print "rad: %f" % rad
                  if rad < np.amin([detector.detector_radius - detector.taper_length, detector.detector_length]):
                      max_val = np.pi/2
                      min_val = 0
                    #   print "theta: min %f pi, max %f pi" % (min_val, max_val)
                  else:
                      if rad < detector.detector_radius - detector.taper_length:
                          #can't possibly hit the taper
                        #   print "less than taper adjustment"
                          min_val = 0
                      elif rad < np.sqrt(detector.detector_radius**2 + detector.taper_length**2):
                          #low enough that it could hit the taper region
                        #   print "taper adjustment"
                          a = detector.detector_radius - detector.taper_length
                          z = 0.5 * (np.sqrt(2*rad**2-a**2) - a)
                          min_val = np.arcsin(z/rad)
                      else:
                          #longer than could hit the taper
                        #   print  " longer thantaper adjustment"
                          min_val = np.arccos(detector.detector_radius/rad)

                      if rad < detector.detector_length:
                          max_val = np.pi/2
                      else:
                          max_val = np.pi/2 - np.arccos(detector.detector_length/rad)
                    #   print "theta: min %f pi, max %f pi" % (min_val, max_val)

                  params[which] += np.pi/2*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], min_val, max_val)

                # if wf_which == 0:
                #     params[which] += (detector.detector_radius)*dnest4.randh()
                #     params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
                elif wf_which == 1:
                    max_val = np.pi/4
                    params[which] += np.pi/4*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], 0, max_val)
                    if params[which] < 0 or params[which] > np.pi/4:
                        print ("wtf phi")

                # elif wf_which == 2:
                #     params[which] += (detector.detector_length)*dnest4.randh()
                #     params[which] = dnest4.wrap(params[which] , 0, detector.detector_length)

                elif wf_which == 3: #scale
                    wf = wfs[wf_idx]
                    min_scale = wf.wfMax - 0.01*wf.wfMax
                    max_scale = wf.wfMax + 0.005*wf.wfMax
                    params[which] += (max_scale-min_scale)*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], min_scale, max_scale)
                #   print "  adjusted scale to %f" %  ( params[which])

                elif wf_which == 4: #t0
                  params[which] += 1*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], min_maxt, max_maxt)
                elif wf_which == 5: #smooth
                  params[which] += 0.1*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], 0, 25)
                #   print "  adjusted smooth to %f" %  ( params[which])

                # elif wf_which == 6: #wf baseline slope (m)
                #     logH -= -0.5*(params[which]/1E-4)**2
                #     params[which] += 1E-4*dnest4.randh()
                #     logH += -0.5*(params[which]/1E-4)**2
                # elif wf_which == 7: #wf baseline incercept (b)
                #     logH -= -0.5*(params[which]/1E-2)**2
                #     params[which] += 1E-2*dnest4.randh()
                #     logH += -0.5*(params[which]/1E-2)**2
                else:
                    print ("wf which value %d (which value %d) not supported" % (wf_which, which))
                    exit(0)

                #   params[which] += 0.01*dnest4.randh()
                #   params[which]=dnest4.wrap(params[which], -1, 1)
                #   print "  adjusted b to %f" %  ( params[which])

        return logH

    # def gen_new_velo_param(self, velo_which, velo_params):
    #             h_100_vlo = vlo_lims[1] - vlo_lims[0]) * rng.rand() + vlo_lims[0]
    #             h_100_vhi = vhi_lims[1] - vhi_lims[0]) * rng.rand() + vhi_lims[0]
    #             h_100_lnbeta = (beta_lims[1] - beta_lims[0]) * rng.rand() + beta_lims[0]
    #
    #     h_100_mu0, h_111_mu0, h_100_lnbeta, h_111_lnbeta, h_111_emu, h_100_mult = velo_params
    #
    #     if velo_which == 0:
    #       h_100_mu0 += (mu0_lims[1] - mu0_lims[0])  *dnest4.randh()
    #       h_100_mu0 = dnest4.wrap(h_100_mu0, mu0_lims[0], mu0_lims[1])
    #     elif velo_which == 1:
    #         h_111_mu0 += (mu0_lims[1] - mu0_lims[0])  *dnest4.randh()
    #         h_111_mu0 = dnest4.wrap(h_111_mu0, mu0_lims[0], mu0_lims[1])
    #     elif velo_which == 2:
    #         h_100_lnbeta += (beta_lims[1] - beta_lims[0])  *dnest4.randh()
    #         h_100_lnbeta = dnest4.wrap(h_100_lnbeta, beta_lims[0], beta_lims[1])
    #     elif velo_which == 3:
    #         h_111_lnbeta += (beta_lims[1] - beta_lims[0])  *dnest4.randh()
    #         h_111_lnbeta = dnest4.wrap(h_111_lnbeta, beta_lims[0], beta_lims[1])
    #     elif velo_which == 4: #h111 e*mu0 param
    #         h_111_emu += (h_111_max_lims[1] - h_111_max_lims[0])  *dnest4.randh()
    #         h_111_emu = dnest4.wrap(h_111_emu, h_111_max_lims[0], h_111_max_lims[1])
    #     elif velo_which == 5: #h100 mult param
    #         h_100_mult += (h_100_mult_lims[1] - h_100_mult_lims[0])  *dnest4.randh()
    #         h_100_mult = dnest4.wrap(h_100_mult, h_100_mult_lims[0], h_100_mult_lims[1])
    #
    #     # h_100_beta = 1./np.exp(h_100_lnbeta)
    #     # h_111_beta = 1./np.exp(h_111_lnbeta)
    #     #
    #     # h_111_e0 = h_111_emu / h_111_mu0
    #     # h_100_e0 = h_100_mult * h_111_emu / h_100_mu0
    #     #
    #     # test_fields = [10, 500,2000]
    #     # is_ok = 1
    #     #
    #     # for field in test_fields:
    #     #     v_100 = find_drift_velocity(field, h_100_mu0, h_100_beta, h_100_e0)
    #     #     v_111 = find_drift_velocity(field, h_111_mu0, h_111_beta, h_111_e0)
    #     #
    #     #     if v_100 < v_111:
    #     #         is_ok = 0
    #     #         break
    #     is_ok = 1
    #     if not is_ok:
    #         return gen_new_velo_param(velo_which, velo_params)
    #     else:
    #         velo_params = [h_100_mu0, h_111_mu0, h_100_lnbeta, h_111_lnbeta, h_111_emu, h_100_mult]
    #         return velo_params
    #

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        b_over_a, c, dc, rc1, rc2, rcfrac, aliasrc = params[tf_first_idx:tf_first_idx+7]
        h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta,  = params[velo_first_idx:velo_first_idx+6]
        # k0_0, k0_1, k0_2, k0_3 = params[k0_first_idx:k0_first_idx+4]
        charge_trapping = params[trap_idx]
        grad = (params[grad_idx])
        avg_imp = (params[imp_avg_idx])

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr,  = params[len(priors):].reshape((6, num_waveforms))

        # print self.changed_wfs
        # print self.ln_likes

        num_changed_wfs = np.sum(self.changed_wfs)
        # print "changed %d waveforms!" % num_changed_wfs

        if num_threads > 1:
            args = []
            for (wf_idx, wf) in enumerate(wfs):
                # if not self.c hanged_wfs[wf_idx]: continue
                # print rad_arr[wf_idx]
                args.append([wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
                              scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
                            #   m_arr[wf_idx], b_arr[wf_idx],
                              b_over_a, c, dc, rc1, rc2, rcfrac,aliasrc,
                              h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta,
                            #   k0_0, k0_1, k0_2, k0_3,
                              grad, avg_imp, charge_trapping, baseline_origin_idx, wf_idx
                            ])

            results = pool.map(WaveformLogLikeStar, args)
            for result in (results):
                self.ln_likes[result['wf_idx']] = result['ln_like']

        else:
            for (wf_idx, wf) in enumerate(wfs):
                # if not self.changed_wfs[wf_idx]: continue
                result = WaveformLogLike(wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
                              scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
                            #   m_arr[wf_idx], b_arr[wf_idx],
                              b_over_a, c, dc, rc1, rc2, rcfrac,aliasrc,
                              h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta,
                            #   k0_0, k0_1, k0_2, k0_3,
                              grad, avg_imp, charge_trapping, baseline_origin_idx, wf_idx
                            )
                self.ln_likes[result['wf_idx']] = result['ln_like']
        return np.sum(self.ln_likes)


def get_velo_params(v_a, v_max, beta):
    E_0 = np.power( (v_max*E_a/v_a)**beta - E_a**beta  , 1./beta)
    mu_0 = v_max / E_0

    return (mu_0,  beta, E_0)


    # E_a = E_lo
    # E_c = E_hi
    #
    # # beta = 1./np.exp(logb)
    #
    # psi = (E_a * v_c) / ( E_c * v_a )
    # E_0 = np.power((psi**beta* E_c**beta - E_a**beta) / (1-psi**beta), 1./beta)
    # mu_0 = (v_a / E_a) * (1 + (E_a / E_0)**beta )**(1./beta)
    #
    # return (mu_0,  beta, E_0)


def WaveformLogLikeStar(a_b):
  return WaveformLogLike(*a_b)

def WaveformLogLike(wf, rad, phi, theta, scale, maxt, smooth,  tf_phi, tf_omega, d, rc1, rc2, rcfrac,aliasrc,
        h_100_va, h_111_va, h_100_vmax, h_111_vmax, h_100_beta, h_111_beta,
        grad, avg_imp, charge_trapping, bl_origin_idx, wf_idx):
    # #TODO: This needs to be length normalized somehow
    # print "think about length normalization, you damn fool"
    # exit(0)

    # print "theta is %f" % (theta/np.pi)
    r = rad * np.cos(theta)
    z = rad * np.sin(theta)

    # tf_d = tf_c * tf_dc
    c = -d * np.cos(tf_omega)
    b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
    a = 1./(1+b_ov_a)
    tf_b = a * b_ov_a

    if np.any(np.less([h_100_va, h_111_va, h_100_vmax, h_111_vmax],0) ):
        return {'wf_idx':wf_idx, 'ln_like':-np.inf}

    h_100_mu0, h_100_beta, h_100_e0 = get_velo_params(h_100_va, h_100_vmax, h_100_beta)
    h_111_mu0, h_111_beta, h_111_e0 = get_velo_params(h_111_va, h_111_vmax, h_111_beta)

    # rc1 = -1./np.log(e_rc1)
    # rc2 = -1./np.log(e_rc2)
    # charge_trapping = -1./np.log(e_charge_trapping)

    # h_100_beta = 1./np.exp(h_100_lnbeta)
    # h_111_beta = 1./np.exp(h_111_lnbeta)

    # h_111_e0 = h_111_emu / h_111_mu0
    # h_100_e0 = h_100_mult * h_111_emu / h_100_mu0

    if scale < 0:
      return  {'wf_idx':wf_idx, 'ln_like':-np.inf}
    if smooth < 0:
       return  {'wf_idx':wf_idx, 'ln_like':-np.inf}
    if not detector.IsInDetector(r, phi, z):
      return  {'wf_idx':wf_idx, 'ln_like':-np.inf}

    detector.SetTransferFunction(tf_b, c, d, rc1, rc2, rcfrac, )
    detector.SetAntialiasingRC(aliasrc)

    detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
    # detector.siggenInst.set_k0_params(k0_0, k0_1, k0_2, k0_3)
    detector.trapping_rc = charge_trapping
    detector.SetGrads(grad, avg_imp)

    data = wf.windowedWf
    model_err = wf.baselineRMS
    data_len = len(data)

    model = detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max", doMaxInterp=doMaxInterp)
    if model is None:
      return  {'wf_idx':wf_idx, 'ln_like':-np.inf}
    if np.any(np.isnan(model)):  return {'wf_idx':wf_idx, 'ln_like':-np.inf}

    # start_idx = -bl_origin_idx
    # end_idx = data_len - bl_origin_idx - 1
    # baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, data_len)
    # model += baseline_trend

    inv_sigma2 = 1.0/(model_err**2)
    ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return  {'wf_idx':wf_idx, 'ln_like':ln_like}


def find_drift_velocity(E, mu_0, beta, E_0, mu_n = 0):
    # mu_0 = 61824
    # beta = 0.942
    # E_0 = 185.
    v = (mu_0 * E) / np.power(1+(E/E_0)**beta, 1./beta) - mu_n*E


def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]

  int_data = int_data[0:max_idx]
  try:
      return np.where(np.less(int_data, percent))[0][-1]
  except IndexError:
      return 0
