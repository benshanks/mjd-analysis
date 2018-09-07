#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *

def main():

    runList = np.arange(11510, 11515)

    mjdList = [
    582,583,580, 581,578, 579,
    692 ,693 ,648, 649 ,640, 641 ,
    610, 610,608, 609, 664, 665,
    #624, 625, 628, 629,688, 689, 694, 695, 614, 615,
    672, 673,
    632, 633,626, 627, 690, 691,
    600, 601, 598, 599,594, 595, 592, 593,
    ]

    #only take high gain channels for now
    chanList = [chan for chan in mjdList if chan%2==0]

    # chanList = [578, 582]

    #data processing

    proc = DataProcessor(detectorChanList=chanList)

    # plt.ion()
    plt.figure()


    plot_waveforms(proc, runList)

def plot_waveforms(proc, runList):
    '''
    settle_time in ms is minimum time since previous event (on same channel)
    '''

    df_bl = pd.read_hdf(proc.channel_info_file_name, key="baseline")
    df_ae = pd.read_hdf(proc.channel_info_file_name, key="ae")

    training_df = []

    for runNumber in runList:
        # df = proc.load_t2(runList)

        t1_file = os.path.join(proc.t1_data_dir,"t1_run{}.h5".format(runNumber))
        t2_file = os.path.join(proc.t2_data_dir, "t2_run{}.h5".format(runNumber))

        df = pd.read_hdf(t2_file,key="data")
        tier1 = pd.read_hdf(t1_file,key=proc.data_key)
        tier1 = tier1.drop({"channel", "energy", "timestamp"}, axis=1)

        df = df.join(tier1, how="inner")

        g4 = dl.Gretina4MDecoder(t1_file)


        for channel, df_chan in df.groupby("channel"):
            if channel != 632: continue

            df_chan = df_chan.dropna()

            # for channel, df_chan in df.groupby("channel"):
            channel = df_chan.channel.unique()[0]
            ae_chan = df_ae.loc[channel]

            avse0, avse1, avse2 = ae_chan.avse0, ae_chan.avse1, ae_chan.avse2
            e = df_chan[proc.ecal_name]
            a_vs_e = df_chan[ae_chan.current_name] - (avse2*e**2 + avse1*e + avse0 )
            df_chan["ae"] = (a_vs_e - ae_chan.avse_mean)/ae_chan.avse_std

            # dt_t0_50 = df_chan.tp_30 - df_chan.tp_10
            # print(dt_t0_50[(df_chan.ae > 0) & (df_chan.isPulser==0) & (df.ecal > 1400)])
            # #
            # plt.hist(df_chan.ae[(df_chan.isPulser==0) & (df.ecal > 1400) & (df.ae > 0)], bins="auto", histtype="step")
            # plt.show()
            # exit()


            df_hiae = df_chan[(df_chan.ae > 75) & (df_chan.isPulser==0)  & (df_chan.ecal > 1400)]

            for i, (index, row) in enumerate(df_hiae.iterrows()):
                wf=g4.parse_event_data(row)
                wf_norm = (wf.get_waveform() - row.bl_int)/row.trap_max
                t0 = int(np.around(row.t0est))
                plt.plot(wf_norm, c="b", alpha=0.1)

                # plt.clf()
                # plt.axvline(t0, c="r")
                # inp = input("uh")
                # if i>25: break

    plt.show()
    exit()


if __name__=="__main__":
    main()
