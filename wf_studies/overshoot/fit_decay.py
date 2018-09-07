#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import pygama.filters as filt
import os
import matplotlib.pyplot as plt
import pygama.decoders as dl
from scipy import optimize
from waffle.models import PulserTrainingModel

def main():

    runList = [848]
    plt.ion()
    f = plt.figure(figsize=(12,9))


    rc_num, rc_den = filt.rc_decay(72)
    overshoot_num, overshoot_den = filt.gretina_overshoot(2, -3.5)

    for runNumber in runList:
        t1_file = "t1_run{}.h5".format(runNumber)
        df = pd.read_hdf(t1_file,key="ORGretina4MWaveformDecoder")
        g4 = dl.Gretina4MDecoder(t1_file)

        chanList = [49]#np.unique(df["channel"])

        for chan in chanList:
            df_chan = df[df.channel == chan]

            for i, (index, row) in enumerate(df_chan.iterrows()):
                if i<10: continue
                wf = g4.parse_event_data(row)
                wf_dat = wf.data - np.mean(wf.data[:200])
                if np.amax(wf_dat) < 200: continue
                if np.count_nonzero( wf_dat > 0.5*wf_dat.max()) < 10: continue

                max_idx = np.argmax(wf_dat>20)

                wf_corr, model, t0 = fit_tail(wf_dat, max_idx)
                plt.plot(wf_dat)
                plt.plot(wf_corr)
                plt.plot(np.arange(max_idx+t0, max_idx+t0+len(model)), model, c="r")
                plt.axvline(max_idx,c="r", ls=":")

                inp = input("q to continue, else to quit")
                if inp == "q": exit()


# def fit_model(wf):
#         wf_conf = {
#             "wf_file_name":wf_file,
#             "wf_idxs":wf_idxs,
#             "align_idx":5,
#             "num_samples":500,
#             "align_percent":5000 #ADC value rather than percentage for pulsers
#         }
#
#         model_conf = [
#             ("OvershootFilterModel",{"zmag_lims":[1, 3]})
#             ]
#
#         conf = FitConfiguration(
#             "",
#             directory = directory,
#             wf_conf=wf_conf,
#             model_conf=model_conf,
#             joint_energy=True,
#             joint_risetime=True,
#             interpType="linear"
#         )
#
#
#     ptm =
#
#     res = optimize.minimize(min_func, [2, -4.5, wf_max, max_idx-20], bounds=[(3,5), (-8,-1), (0.8*wf_max, 1.2*wf_max), (225, 275)])
#

def fit_tail(wf_data, max_idx):
    '''
    try to fit out the best tail parameters to flatten the top
    '''

    fit_offset=-20

    # wf_data[260:265] =13565
    tail_data = wf_data[max_idx+fit_offset:]

    # plt.figure()
    # plt.plot(wf_data, c="k")
    # plt.plot(np.arange(len(tail_data))+max_idx+fit_offset, tail_data)
    # inp = input("")
    # exit()

    model_err = np.std(wf_data[:200])

    square = np.zeros( (len(wf_data) + 120) )
    square[20:] = 1

    def digital_filter_model(x):
        overshoot_decay, overshoot_pole_rel, energy= x[:3]
        overshoot_num, overshoot_den = filt.gretina_overshoot(overshoot_decay, overshoot_pole_rel)
        square_proc = signal.lfilter(overshoot_num, overshoot_den, square)
        model = square_proc*energy*1000

        return model

    def get_model(x):
        t0 = x[-1]
        t0_off = t0 - np.floor(t0)

        model = digital_filter_model(x)

        if t0_off == 0:
            model_interp = model
        else:
            model_interp= np.interp(np.arange(1, len(square))-t0_off, np.arange(len(square)), model )

        model_comp = np.zeros_like(wf_data)
        t0_ceil = int(np.ceil(t0))

        model_comp[t0_ceil:] = model_interp[:len(wf_data)-t0_ceil]

        # plt.figure()
        # plt.plot(wf_data, c="k")
        # plt.plot(model_comp)
        # inp = input("")
        # exit()


        return model_comp

    def min_func(x):
        t0 = x[-1]
        if t0<250 or t0 > 255: return np.inf
        # model = get_model(x)
        model = overshoot_curve(*x)

        inv_sigma2 = 1.0/(model_err**2)

        model_tail = model[max_idx+fit_offset:]

        # plt.figure()
        # plt.plot(wf_data, c="k")
        # plt.plot(np.arange(len(tail_data))+max_idx+fit_offset, tail_data)
        # plt.plot(model)
        # plt.plot(np.arange(len(tail_data))+max_idx+fit_offset, model_tail)
        # inp = input("")
        # exit()

        return 0.5*(np.sum((tail_data-model_tail)**2*inv_sigma2 - np.log(inv_sigma2)))


    def overshoot_curve(a,b,r,energy, t0):
        t = np.arange(len(wf_data))
        model = np.zeros_like(wf_data)

        model[t<t0] = 0
        t0_off = t0 - np.floor(t0)

        def curve(t_in, overfrac,b,r,energy):
            a = b/overfrac
            t = t_in*10/1E3
            return  energy*(-b*-r*-r/-a)*(np.exp(r*t)*(-a*b +2*a*r-r**2)/(r**2*(b-r)**2)
                +   a/(b*r**2)
                +(t * (a-r)*np.exp(r*t))/(r*(b-r))
                +(b-a)*np.exp(b*t)/(b*(b-r)**2))

        t_pos = t[t>=t0]
        if t0_off == 0:
            model[t>=t0] = curve(t_pos, a,b,r,energy*1000)
        else:
            t_mod = t_pos-t0_off
            model[t>=t0] = curve(t_mod, a,b,r,energy*1000)

        # plt.figure()
        # plt.plot(wf_data, c="k")
        # plt.plot(model)
        # inp = input("")
        # exit()

        return model


    def overshoot_david(t, a,r,e):
            return 1000*e*(1 + r*np.exp(-t*a))

        # return 1000*energy*((pole-zero)*np.exp(pole*t)/(pole*(pole-lp_rc))
        #         +   (zero-lp_rc)*np.exp(lp_rc*t)/(lp_rc*(pole-lp_rc))
        #         - zero/(pole*lp_rc))


    wf_max = 13550/1000
    print(wf_max)

    # res = optimize.minimize(min_func, [5, -4.5, wf_max, max_idx-20], bounds=[(3,8), (-8,-1), (0.8*wf_max, 1.2*wf_max), (225, 275)])
    # model = get_model(res["x"])

    bounds = [(1.001, 1.1), (-1,0), (-np.inf,0), (0, 1.1*wf_data[-1]/1000), (max_idx-2, max_idx+2)  ]
    res = optimize.minimize(min_func, [1.05, -0.45, -10,wf_data[-1]/1000, max_idx-1],bounds=bounds, method="Nelder-Mead" )
    model = overshoot_curve(*res["x"])

    print(res["x"])
    a,b,r,energy,t0 = res["x"]

    plt.figure()
    plt.plot(wf_data, c="k")
    plt.plot(np.arange(len(tail_data))+max_idx+fit_offset, tail_data)

    plt.plot(model)
    # for i in [0.5, 0.75, 1, 1.25, 1.5]:
    #     model2 = overshoot_curve(a,b,i*r,energy,t0)
    #     plt.plot(model2)

    plt.plot(np.arange(len(tail_data))+max_idx+fit_offset, model[max_idx+fit_offset:])

    inp = input("")
    exit()


    # p0 = [2, .001, wf_max]
    # t = np.arange(fit_offset, len(tail_data)+fit_offset)*10/1E3
    # plt.figure()
    # plt.plot(t, overshoot_david(t, *p0))
    # inp = input("fdgs")
    # exit()

    # p, _ = optimize.curve_fit(overshoot_david, t, tail_data,
    # p0, sigma=model_err*np.ones_like(tail_data), method="dogbox",
    # bounds = [(0,0, 0),(10,0.1,np.inf)]
    # )
    # rc2,f,e = p
    # print(p)
    # model = overshoot_david(np.arange(0, len(tail_data)+fit_offset)*10/1E3, *p)
    # wf_proc = np.nan

    overshoot_num, overshoot_den = filt.gretina_overshoot(rc2, f)
    wf_proc = signal.lfilter(overshoot_den, overshoot_num, wf_data)
    #

    return wf_proc, model, t0


if __name__=="__main__":
    main()
