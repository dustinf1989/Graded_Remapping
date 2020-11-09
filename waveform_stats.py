#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

#hdf5Dir = '/home/dustin/Documents/hdf5_v1/' # Load data from this folder
hdf5Dir = '/home/fetterhoff/atlas/RM_45max/combined_hdf5_from_raw_gsp2_removedInfSR/'

combinedResultDir = hdf5Dir+'waveform_stats/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

wavedf = pd.read_csv(hdf5Dir+'allwavecombined_final.csv')
wavedf['neuron_type'] = 0
fil=(wavedf.spike_ratio < 1.5)
wavedf['neuron_type'].loc[fil] = 'INT'
fil=(wavedf.spike_ratio >= 1.5)
wavedf['neuron_type'].loc[fil] = 'PC'


#%% scatter spike ratio vs firing rate

pl.rcParams.update({'font.size': 6})

ajp = sns.jointplot(x=wavedf.spike_ratio[wavedf.spike_ratio < 10],y=np.log10(wavedf.mfr[wavedf.spike_ratio < 10]),alpha=0.25,joint_kws={"s": 1}, height=2.2)
yl1, yl2 = ajp.ax_joint.get_ylim()
ajp.ax_joint.fill_between([0,1.5],yl1, yl2, facecolor='C1',alpha=0.2)
ajp.ax_joint.fill_between([1.5,10],np.log10(5.0), yl2, facecolor='C1',alpha=0.2)
ajp.ax_joint.set_ylim(yl1, yl2)
ajp.ax_joint.set_xlabel('Spike Ratio')
ajp.ax_joint.set_ylabel('Log of Mean Firing Rate (Hz)')
ajp.ax_marg_x.axvline(1.5,color='C1',alpha=0.5)
ajp.ax_joint.set_ylim(yl1, yl2)
ajp.savefig(combinedResultDir+'fig_S1B_mfr_spike_ratio.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
pl.close()
perc_lost = (wavedf.spike_ratio > 10).sum() / float(len(wavedf)) *100
print '{}% of neurons not shown (SR > 10)'.format(perc_lost)

#%%

ax = sns.catplot(x='neuron_type', y='spike_width_ms', kind='boxen', data=wavedf, height=2.2)
ax.set_ylabels('Spike Width (ms)')
ax.set_xlabels('Neuron Type')
ax.savefig(combinedResultDir+'fig_S1C_spike_width.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
pl.close()
