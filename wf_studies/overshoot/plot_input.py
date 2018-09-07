#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df = pd.read_csv("5ns_rt_zoom.txt", header=None)
    print (df.head())
    plt.plot(df[0]*1E9,df[1])
    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )
