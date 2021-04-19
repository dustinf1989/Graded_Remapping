#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script generates all panels of Figure S2 in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

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

fileList = fileList
speedThresh = 5 # cm/s, to discard spikes during stillness

hdf5Dir = '/home/dustin/Documents/data/combined_hdf5_from_raw_revision_v3/'
combinedResultDir = hdf5Dir+'trajectory_plots/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

bd = [187.5, 275, 412.5, 500] # boundaries for all maze segments

totalMazeLength = 622 # measured from the setup
pl.rcParams.update({'font.size': 6, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6, 'axes.facecolor':'white', 'lines.linewidth': 1.0, 'lines.markersize': 2.0})
maze_type_color_dict = {1:'r',-1:'b',2:'m',-2:'c'}
numToTypeDict = {0 : 'None', 1: 'R', 2:'L',3:'R*', 4:'L*', 5:'LR*-im', 6:'LL*-dir', 7:'swap*', 8:'og', 9:'RR*-dir',10:'RL*-im',11:'3pk',12:'3pk',13:'3pk',14:'3pk',15:'4pk'}

minWindow = 4 # Number of adjacent peak bins to keep cell
session_list, maze_seg_code = [], []
toPlotAllNeurons = False

all_vr_y = np.array([])
all_speed = np.array([])
all_places_cm = np.array([])
all_maze_type = np.array([])

#%% Load data for each session
for il, s in enumerate(fileList):
    session = s[0]
    print(session) # current session

    sd = hdf5Dir+session+'/' # session directory

    # Load the necessary files
#    f1 = hdf5Dir+session+'_dat.h5'
#    spikeDF = pd.read_hdf(f1, 'spikeDF')

    f2 = sd+session+'_laps_traj.h5'
    lapsDF = pd.read_hdf(f2, 'lapsDF')
    trajDF = pd.read_hdf(f2, 'trj')
    lapsDB = np.array(lapsDF)

#    f3 = hdf5Dir+session+'_resultsDB.h5'
#    cellResultsDB = pd.read_hdf(f3, 'cellResultsDB')

    #%% Plot trajectory for each lap
    fig, ax = pl.subplots(4,10,figsize=(6.65,2.5), sharex=True, sharey=True)
    fig.tight_layout(pad = 1.0)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    pl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pl.xlabel("virtual x-position")
    pl.ylabel("virtual y-position",labelpad=12)

    ai, bi, ci, di = 0, 0,0,0

    ax[0,0].set_ylabel('R'), ax[1,0].set_ylabel('L')
    ax[2,0].set_ylabel('R*'), ax[3,0].set_ylabel('L*')

    rowsDict = {1:0, -1:1, 2:2, -2:3}

    places_xy = np.array([trajDF.vr_x, trajDF.vr_y]).copy()
    places_xy[1,:] -= 4.5 # Center places for plotting
    laps = np.unique(trajDF.lap_num)[np.unique(trajDF.lap_num) > -1].astype(int)

    for i in laps:
        # based on "real" positions
        lapmask = np.array(trajDF.lap_num == i)
        maze_type = np.unique(trajDF.maze_type[lapmask])[0]
        it = rowsDict[maze_type] # Row index
        fhr = (trajDF.places_cm < bd[0]) & lapmask
        fcr = (trajDF.places_cm >= bd[0]) & (trajDF.places_cm < bd[1]) & lapmask
        mhr = (trajDF.places_cm >= bd[1]) & (trajDF.places_cm < bd[2]) & lapmask
        lcr = (trajDF.places_cm >= bd[2]) & (trajDF.places_cm < bd[3]) & lapmask
        lhr = (trajDF.places_cm >= bd[3]) & lapmask

        if maze_type == 1:
            qi = ai
            ai += 1
        if maze_type == -1:
            qi = bi
            bi += 1
        if maze_type == 2:
            qi = ci
            ci += 1
        if maze_type == -2:
            qi = di
            di += 1

        if qi < 10:
            ax[it,qi].plot(places_xy[0,fhr], places_xy[1,fhr],'C1'),
            ax[it,qi].plot(places_xy[0,fcr], places_xy[1,fcr], 'C8'),
            ax[it,qi].plot(places_xy[0,mhr], places_xy[1,mhr], 'C2'),
            ax[it,qi].plot(places_xy[0,lcr], places_xy[1,lcr],'C6'),
            ax[it,qi].plot(places_xy[0,lhr], places_xy[1,lhr],'C4'),
            ax[it,qi].set_yticks([-4,0,4])
            ax[it, qi].set_title('lap '+str(i+1),pad=2,fontsize=6)

            if np.logical_or(it==0,it==2):
                ax[it,qi].plot([0,0],[.5,-0.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([0,7],[-.5,-0.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([0,7],[.5,0.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([7,9],[-.5,-2.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([7,9],[.5,-1.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([9,15],[-2.5,-2.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([9,15],[-1.5,-1.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([15,17],[-2.5,-4.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([15,17],[-1.5,-3.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([17,24],[-4.5,-4.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([17,24],[-3.5,-3.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([24,24],[-3.5,-4.5],'k',alpha=1.0,linewidth=0.5)
            if np.logical_or(it==1,it==3):
                ax[it,qi].plot([0,0],[.5,-0.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([0,7],[-.5,-0.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([0,7],[.5,0.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([7,9],[-.5,1.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([7,9],[.5,2.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([9,15],[2.5,2.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([9,15],[1.5,1.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([15,17],[2.5,4.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([15,17],[1.5,3.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([17,24],[4.5,4.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([17,24],[3.5,3.5],'k',alpha=1.0,linewidth=0.5)
                ax[it,qi].plot([24,24],[3.5,4.5],'k',alpha=1.0,linewidth=0.5)

    ax[3,1].legend(['First Hall','First Corner','Middle Hall','Last Corner','Last Hall'])

    pl.savefig(combinedResultDir+'fig_S2A_lap_trajectories_{}.pdf'.format(session),format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.01)
    pl.close()

    all_vr_y = np.append(all_vr_y, places_xy[1,:] * -1)
    all_speed = np.append(all_speed, trajDF.speed)
    all_places_cm = np.append(all_places_cm, trajDF.places_cm)
    all_maze_type = np.append(all_maze_type, trajDF.maze_type)

#%% y-Position histograms
vr_y = places_xy[1,:] * -1 # Mirror the axis so it plots in an order than makes sense
spmask = np.array(all_speed > speedThresh)
fhr = (all_places_cm < bd[0]) & spmask
mhr = (all_places_cm >= bd[1]) & (all_places_cm< bd[2]) & spmask
lhr = (all_places_cm >= bd[3]) & spmask

fig, ax = pl.subplots(1,3, figsize=(3.5,1.6),sharey=True)
ax = ax.ravel()
fig.suptitle('Histograms of virtual y-position by maze segment',y=1.1)
fig.tight_layout()

dictnc = {1:'r', -1:'b', 2:'m', -2:'c'}

for mt,c in dictnc.items():
    bins1 = np.linspace(-0.5,0.55,21)
    hi, _ = np.histogram(all_vr_y[fhr & (all_maze_type==mt)],bins1,density=False)
    ax[0].plot(bins1[:-1],hi.astype(float)/hi.sum(),c)

    if (mt == 1) or (mt == 2):
        bins = np.linspace(1.5,2.5,21)
    elif (mt == -1) or (mt == -2):
        bins = np.linspace(-2.5,-1.5,21)

    hi, _ = np.histogram(all_vr_y[mhr & (all_maze_type==mt)],bins,density=False)
    ax[1].plot(bins1[:-1],hi.astype(float)/hi.sum(),c)

    if (mt == 1) or (mt == 2):
        bins = np.linspace(3.5,4.5,21)
    elif (mt == -1) or (mt == -2):
        bins = np.linspace(-4.5,-3.5,21)

    hi, _ = np.histogram(all_vr_y[lhr & (all_maze_type==mt)],bins,density=False)
    ax[2].plot(bins1[:-1],hi.astype(float)/hi.sum(),c)

for i, nm in enumerate(['First hall', 'Middle hall', 'Last hall']):
    ax[i].set_xlabel('virtual y-position')
    ax[i].set_title(nm)
#    ax[i].set_xticks([-0.5,0.5], ('Left','Right'))
ax[0].set_ylabel('PDF')
pl.setp(ax, xticks=[-0.5,0.5], xticklabels=['Left','Right'])

pl.savefig(combinedResultDir+'fig_S2C_y_position_hist.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.01)
pl.close()

#%% average speed by maze segment

spmask = np.array(all_speed > speedThresh)
fhr = (all_places_cm < bd[0]) & spmask
fcr = (all_places_cm >= bd[0]) & (all_places_cm < bd[1]) & spmask
mhr = (all_places_cm >= bd[1]) & (all_places_cm < bd[2]) & spmask
lcr = (all_places_cm >= bd[2]) & (all_places_cm < bd[3]) & spmask
lhr = (all_places_cm >= bd[3]) & spmask

pl.figure(figsize=(1.5,1.4))
x = [0, 1, 2, 3, 4]
for mt,c in dictnc.iteritems():
    mm = all_maze_type == mt
    sp_mean = np.array([all_speed[fhr & mm].mean(axis=0), all_speed[fcr & mm].mean(axis=0), all_speed[mhr & mm].mean(axis=0), all_speed[lcr & mm].mean(axis=0), all_speed[lhr & mm].mean(axis=0)])
    sp_std = np.array([all_speed[fhr & mm].std(axis=0), all_speed[fcr & mm].std(axis=0), all_speed[mhr & mm].std(axis=0), all_speed[lcr & mm].std(axis=0), all_speed[lhr & mm].std(axis=0)])
    pl.plot(x, sp_mean,ls=':',color=c)

pl.legend(['R','L','R*','L*'])
pl.xticks(np.arange(5), ('First Hall', 'First Corner', 'Middle Hall', 'Last Corner', 'Last Hall'),rotation=45)
pl.ylabel('Average Speed (cm/s)')

pl.savefig(combinedResultDir+'fig_S2B_avg_speed_by_maze_segment.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
pl.close()