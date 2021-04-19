#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script generates panels B-I of Figure 1 and S3 in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import scipy.stats as ss

#inPath = '/home/dustin/Documents/'
inPath = '/media/dustin/easystore/RM_45max/'

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

#hdf5Dir = '/home/dustin/Documents/hdf5_v1/' # Load data from this folder
#hdf5Dir = '/home/fetterhoff/ownCloud/Documents/manuscript_data/'
hdf5Dir = '/home/fetterhoff/atlas/RM_45max/combined_hdf5_from_raw_gsp2_removedInfSR/'

combinedResultDir = hdf5Dir+'place_cells/' # Save in subdirectory
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
sra = np.array([])

#%% Load data for each session
for il in np.arange(len(fileList)):
    folderName = fileList[il]
    session = folderName[0][-9:]
    print(session)

    # Load the necessary files
    f1 = hdf5Dir+session+'_dat.h5'
    spikeDF = pd.read_hdf(f1, 'spikeDF')

    f3 = hdf5Dir+session+'_resultsDB.h5'
    cellResultsDB = pd.read_hdf(f3, 'cellResultsDB')

    sra = np.append(sra, spikeDF.spike_ratio)

    #%% Plot neurons as examples
    for q, cell_id in enumerate(spikeDF.T):
        sp = spikeDF.loc[cell_id]
        title = combinedResultDir + session + '_' + sp.file[:-2]
        session_list.append(session)
        maze_seg_code.append(sp.segment_types)
        if toPlotAllNeurons:
            sp.segment_type_cat = []
            for i in sp.segment_types:
                sp.segment_type_cat.append(numToTypeDict[i])

            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == cell_id]
            sp.run_real_cm = sp.real_cm[(sp.speed_cms > speedThresh) & (sp.lap_num > -1)]
            sp.run_lap_num = sp.lap_num[(sp.speed_cms > speedThresh) & (sp.lap_num > -1)]
            sp.run_maze_type = sp.maze_type[(sp.speed_cms > speedThresh) & (sp.lap_num > -1)]

            fig, ax = pl.subplots(2,1, figsize=(2.1,2.1), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
            ax = ax.ravel()

            for mt, c in maze_type_color_dict.iteritems():
                ax[0].plot(sp.run_real_cm[sp.run_maze_type==mt], sp.run_lap_num[sp.run_maze_type==mt]+1,'|', color=c, alpha = 0.5)

            ax[0].set_ylabel('Lap #')
            ax[0].set_ylim([0.5,40.6])
            ax[0].set_title('{}'.format(' - '.join(sp.segment_type_cat)))

            for color, (mt, da) in zip(['c', 'b', 'r', 'm'], resultsDB.groupby('choice')):
                fieldWidth = np.sum(da.PeakWidthFinal)
                da.plot.line(x='XpositionReal', y='normRate', c=maze_type_color_dict[mt], ls='-', marker='.', ax=ax[1], markersize=2)

                if fieldWidth >= minWindow:
                    ax[1].fill_between(da.XpositionReal, 0, ax[1].get_ylim()[1], where=da.PeakWidthFinalAboveT == 1, facecolor=maze_type_color_dict[mt], alpha=0.2)

            ax[1].set_xlim([0,totalMazeLength])
            ax[1].set_ylim([0,resultsDB.normRate.max()*1.1])
            ax[1].set_xlabel('Track Position (cm)')
            ax[1].set_ylabel('Firing rate (Hz)')
            ax[1].get_legend().remove()

            for i in range(2):
                y1, y2 = ax[i].get_ylim()
                ax[i].fill_between([bd[0],bd[1]],y1,y2,facecolor='k',alpha=0.1)
                ax[i].fill_between([bd[2],bd[3]],y1,y2,facecolor='k',alpha=0.1)
                ax[i].set_xlim([0,totalMazeLength])

            ax[1].set_ylim([0,y2])
            ax[1].set_xticks([0,200,400,600])

            fig.savefig(title+'_{}.pdf'.format(sp.segment_type_cat[0]),format='pdf',dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            pl.close(fig)

#%% Pie chart
maze_seg_code = np.squeeze(np.array(maze_seg_code))
typebinedges = np.arange(-.5,17.5,1)
data0 = np.zeros([5,len(typebinedges)-2])
segment_names = ['First Hallway', 'First Corner', 'Middle Hallway', 'Last Corner', 'Last Hallway']
labels = ['Single maze','Original / Swap','Image','Direction','3-4 Peaks']
cl = ['C7','C8','C4','C1','C2']
barWidth = 0.85
r = [0,1,2,3,4]

fig, ax = pl.subplots(2,3, figsize=(8.4,5.8))
ax = ax.ravel()
pl.subplots_adjust(wspace=0, hspace=0)

for i in range(5):
    output1 = np.histogram(maze_seg_code[:,i], typebinedges)[0]
    data0[i] = output1[1:]
    single_maze_total = np.sum(output1[1:5])
    old_new_total = np.sum(output1[7:9])
    im_total = output1[5] + output1[10]
    dir_total = output1[6] + output1[9]
    peaks34_total = np.sum(output1[11:])

    sizes = [single_maze_total, old_new_total, im_total, dir_total, peaks34_total]
    ax[i].pie(sizes,colors=cl, autopct='%1.1f%%')
    ax[i].set_title(segment_names[i], pad=-10)

    if i==0:
        im_fh, rest_fh = im_total, output1[1:].sum()-im_total
    elif i==2:
        im_mh, rest_mh = im_total, output1[1:].sum()-im_total
    elif i==4:
        im_lh, rest_lh = im_total, output1[1:].sum()-im_total

# Are there more image cells in the middle or last hallways compared to random (the first hallway)?
ss.chisquare([im_fh, rest_fh], [im_mh, rest_mh])
ss.chisquare([im_fh, rest_fh], [im_lh, rest_lh])

ax[1].legend(labels, loc='upper left', bbox_to_anchor=(0.3,0.83))

# everything shifted one back because first row is removed
df = pd.DataFrame({'RR*-dir' : data0[:,8], 'LL*-dir' : data0[:,5], 'RL*-im' : data0[:,9], 'LR*-im' : data0[:,4]})

# From raw value to percentage
totals = [i+j+k+l for i,j,k,l in zip(df['RR*-dir'], df['LL*-dir'], df['RL*-im'], df['LR*-im'])]
bars1 = [i for i in df['RR*-dir']]
bars2 = [i for i in df['LL*-dir']]
bars3 = [i for i in df['RL*-im']]
bars4 = [i for i in df['LR*-im']]

# Create stacked bar graph
ax[5].bar(r, bars1, color='C1', edgecolor='white', width=barWidth)
ax[5].bar(r, bars2, bottom=bars1, color='C1', edgecolor='white', width=barWidth, alpha=0.8)
ax[5].bar(r, bars3, bottom=[i+j for i,j in zip(bars1, bars2)], color='C4', edgecolor='white', width=barWidth)
ax[5].bar(r, bars4, bottom=[i+j+k for i,j,k in zip(bars1, bars2, bars3)], color='C4', edgecolor='white', width=barWidth, alpha=0.8)

pl.sca(ax[5])
pl.xticks(r, segment_names, rotation=60)
ax[5].set_ylabel('Number of neurons')
ax[5].spines['right'].set_visible(False)
ax[5].spines['top'].set_visible(False)
ax[5].legend(['RR*-dir','LL*-dir','RL*-im','LR*-im'], loc='upper right')
ax[5].set_xlim([-0.5,4.5])
ax[5].set_ylim([0,95])
ax[5].set_yticks(np.arange(0,96,5), minor=True)
box = ax[5].get_position()
box.x0, box.y0, box.y1 = 0.7, 0.26, 0.5
ax[5].set_position(box)

pl.savefig(combinedResultDir+'fig_1HI_pieChart.pdf',format='pdf', bbox_inches = 'tight', pad_inches = 0.01)
pl.close()

#%% Pie for each gerbil
session_list = np.array(session_list)
gerbil_names = ['0395', '0397', '2017', '2018', '2783', '2784']

for gid in gerbil_names:
    gloc = np.where(np.char.find(session_list, gid)>=0)[0]

    fig, ax = pl.subplots(2,3, figsize=(4.0,2.7))
    fig.suptitle('Gerbil {}'.format(gid), y=0.95,fontsize=8)
    ax = ax.ravel()
    pl.subplots_adjust(wspace=0, hspace=0)

    for i in range(5):
        output1 = np.histogram(maze_seg_code[gloc,i], typebinedges)[0]
        data0[i] = output1[1:]
        single_maze_total = np.sum(output1[1:5])
        old_new_total = np.sum(output1[7:9])
        im_total = output1[5] + output1[10]
        dir_total = output1[6] + output1[9]
        peaks34_total = np.sum(output1[11:])

        sizes = [single_maze_total, old_new_total, im_total, dir_total, peaks34_total]

        ax[i].pie(sizes, colors=cl)
        ax[i].set_title(segment_names[i], pad=-5)

    # 5=LL*-dir, 9=RL*-im, 8=RR*-dir, 4=LR*-im # everything shifted one back because first row is removed
    df = pd.DataFrame({'RR*-dir' : data0[:,8], 'LL*-dir' : data0[:,5], 'RL*-im' : data0[:,9], 'LR*-im' : data0[:,4]})

    # From raw value to percentage
    totals = [i+j+k+l for i,j,k,l in zip(df['RR*-dir'], df['LL*-dir'], df['RL*-im'], df['LR*-im'])]
    bars1 = [i for i in df['RR*-dir']]
    bars2 = [i for i in df['LL*-dir']]
    bars3 = [i for i in df['RL*-im']]
    bars4 = [i for i in df['LR*-im']]

    # Create green Bars
    ax[5].bar(r, bars1, color='C1', edgecolor='white', width=barWidth)
    ax[5].bar(r, bars2, bottom=bars1, color='C1', edgecolor='white', width=barWidth, alpha=0.8)
    ax[5].bar(r, bars3, bottom=[i+j for i,j in zip(bars1, bars2)], color='C4', edgecolor='white', width=barWidth)
    ax[5].bar(r, bars4, bottom=[i+j+k for i,j,k in zip(bars1, bars2, bars3)], color='C4', edgecolor='white', width=barWidth, alpha=0.8)

    pl.sca(ax[5])
    pl.xticks(r, segment_names, rotation=60)
    ax[5].set_ylabel('Number of neurons',labelpad=1)
    ax[5].spines['right'].set_visible(False)
    ax[5].spines['top'].set_visible(False)
    ax[5].set_xlim([-0.5,4.5])
    ax[5].set_ylim([0,25])
    ax[5].set_yticks(np.arange(0,26,2), minor=True)
    box = ax[5].get_position()
    box.x0, box.y0, box.y1 = 0.7, 0.26, 0.5
    ax[5].set_position(box)

    if gid == '0395':
        ax[5].legend(['RR*-dir','LL*-dir','RL*-im','LR*-im'], loc='upper right')
        ax[1].legend(labels, loc='upper left', bbox_to_anchor=(0.3,0.83))

    pl.savefig(combinedResultDir+'fig_S2_pieChartByGerbil_{}.pdf'.format(gid),format='pdf', bbox_inches = 'tight', pad_inches = 0.01)
    pl.close()