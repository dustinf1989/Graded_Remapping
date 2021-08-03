#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script generates Figure S1B in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.

All analysis code was written by D. Fetterhoff

"""
import os
import glob
import numpy as np
import matplotlib.pyplot as pl
from scipy.io import loadmat

fileList = [
    ['g0395_d1'],
    ['g0395_d2'],
    ['g0395_d3'],
    ['g0397_d1'],
    ['g0397_d2'],
    ['g0397_d3'],
    ['g2017_d1'],
    ['g2017_d2'],
    ['g2017_d3'],
    ['g2018_d1'],
    ['g2018_d2'],
    ['g2018_d3'],
    ['g2783_d1'],
    ['g2783_d2'],
    ['g2783_d3'],
    ['g2784_d1'],
    ['g2784_d2'],
    ['g2784_d3']
    ]

# Load data from this folder
hdf5Dir = '/home/fetterhoff/Documents/graded_remapping_data/Graded_Remapping/'

combinedResultDir = hdf5Dir+'waveform_stats/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

pl.rcParams.update({'font.size': 6, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6, 'axes.facecolor':'white', 'lines.linewidth': 1.25, 'lines.markersize': 2.0, 'axes.labelsize': 6, 'figure.titlesize' : 6, 'axes.titlesize' : 'medium'})

iso_dist= np.array([])

#%%
for il, s in enumerate(fileList):
    session = s[0]
    print(session) # current session

    sd = hdf5Dir+session+'/' # session directory

    for mat_name in glob.glob(sd+'*TT*.mat'): # loop through all neuron files
        m = loadmat(mat_name)
        iso_dist = np.append(iso_dist, m['isolation_distance'][0][0]) # save isolation distances

#%% Plot cumulative distribution of Isolation Distance

pl.figure(figsize=(2.,1.5))
bins = np.arange(-0.1,100,.1)

ht, _ = np.histogram(iso_dist,bins)
cum = np.cumsum(ht).astype(float) / np.cumsum(ht).max()

pl.plot(bins[:-1], cum)

pl.xlabel('Isolation Distance')
pl.ylabel('Cumulative Distribution')
pl.xlim([0,100])

pl.savefig(combinedResultDir+'Fig_S1B_IsolationDistance.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
pl.close()
