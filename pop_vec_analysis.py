#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script generates all panels of Figure 4, 5A and S5 in the manuscript
"Graded Remapping of Hippocampal Ensembles under Sensory Conflicts" written by
D. Fetterhoff, A. Sobolev & C. Leibold.

To generate Fig. 4B without image cells, set toExcludeImageCells to True.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import itertools as it
import pandas as pd

fileList = [
    ['g0395_d1'],
    ['g0395_d2'],
    ['g0395_d3'],
    ['g0397_d1'],
    ['g0397_d2'],
    ['g0397_d3'], #5
    ['g2017_d1'],
    ['g2017_d2'],
    ['g2017_d3'],
    ['g2018_d1'],
    ['g2018_d2'], #10
    ['g2018_d3'],
    ['g2783_d1'],
    ['g2783_d2'], #13
    ['g2783_d3'],
    ['g2784_d1'],
    ['g2784_d2'],
    ['g2784_d3'] #17
    ]

# Select sessions with the most place cells
fileList6 = np.array(fileList)[[0,5,6,10,13,17]]

fileListSim = [
      ['g0397_ss1'],
      ['g0397_ss2'],
      ['g2018_ss1'],
      ['g2783_ss1'],
      ['g2783_ss2'],
      ['g2784_ss1']]

#hdf5Dir = '/home/dustin/Documents/hdf5_v1/' # Load data from this folder
#hdf5Dir = '/home/fetterhoff/atlas/RM_45max/combined_hdf5_from_raw_gsp2_removedInfSR/'
hdf5Dir = '/home/fetterhoff/atlas/VR_Data/combined_hdf5_from_raw_gsp/'

combinedResultDir = hdf5Dir+'pop_vec_analysis/' # Save in subdirectory
if not os.path.exists(combinedResultDir):
    os.makedirs(combinedResultDir)

toExcludeImageCells = False # Remove image cells, set True for Fig. 4B
doITI = True # Set True for Fig. 5A
simSwap = True # Set True for Fig. S5A-B
best6 = False # Set True for Fig. S5C-D

if simSwap:
    fileList = fileListSim
if best6:
    fileList = fileList6
else:
    fileList = fileList

# Need to double check this
totalMazeLength = 622 # measured from the setup
xBins = np.linspace(1,24,80) # old: np.arange(1,24,0.29) np.linspace(1,24,80)
xCenters = (xBins + np.diff(xBins)[0]/2)[:-1]
Nbins = xBins.size -1
xBinsReal = np.linspace(0, totalMazeLength, len(xBins)) #np.linspace(7.94871795, totalMazeLength, 80) # Not sure why we start at 7.9?
xCentersReal = (xBinsReal + np.diff(xBinsReal)[0]/2)[:-1]

bd = [187.5, 275, 412.5, 500] # boundaries for all maze segments
ibd = [np.abs(xCentersReal - b).argmin() for b in bd] # index of boundaries
itmp = ibd[:]
itmp.insert(0,0), itmp.append(Nbins-1)
iparts = [i+1 for i in itmp]

mazeTypeList = ['R', 'L', 'R*', 'L*']

#%% Initialize vectors to get values over all sessions
df_count = pd.DataFrame()
correlations_mat_sessions = np.zeros([len(fileList),16,Nbins])
correlations_shuf_sessions = np.zeros([len(fileList),16,Nbins])

iti_correlations_mat_sessions = np.zeros([len(fileList),16,200])
iti_correlations_shuf_sessions = np.zeros([len(fileList),16,200])

pl.rcParams.update({'font.size': 6, 'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6, 'axes.facecolor':'white', 'lines.linewidth': 0.75, 'lines.markersize': 2.0, 'axes.labelsize': 6, 'figure.titlesize' : 6, 'axes.titlesize' : 'medium'})

#%% Loop through all sessions
for il in np.arange(len(fileList)):
    folderName = fileList[il]
    session = folderName[0][-9:]
    print(session)

    # Load the necessary files
    f1 = hdf5Dir+session+'_dat.h5'
    spikeDF = pd.read_hdf(f1, 'spikeDF')

    f2 = hdf5Dir+session+'_laps_traj.h5'
    lapsDF = pd.read_hdf(f2, 'lapsDF')
    trajDF = pd.read_hdf(f2, 'trj')
    lapsDB = np.array(lapsDF)

    f3 = hdf5Dir+session+'_resultsDB.h5'
    cellResultsDB = pd.read_hdf(f3, 'cellResultsDB')
    f4 = hdf5Dir+session+'_itiResultsDB.h5'
    itiResultsDB = pd.read_hdf(f4, 'itiResultsDB')

    nPlaceFields = 0 # Count the number of place fields
    for i in spikeDF.FieldPeakLoc:
        nPlaceFields += len(i)

    # Table S1
    sumN = pd.DataFrame({'session': session, 'nPlaceCells' : len(spikeDF), 'nPlaceFields' : nPlaceFields}, index=[il])
    df_count = pd.concat([df_count, sumN])

    maze_seg_code = []
    for q, cell_id in enumerate(spikeDF.T):
        maze_seg_code.append(spikeDF.loc[cell_id].segment_types)
    maze_seg_code = np.squeeze(np.array(maze_seg_code))

    if toExcludeImageCells:
        simple_pc_bool = np.logical_or(maze_seg_code == 5, maze_seg_code == 10).sum(axis=1) < 1 # Determine image cells
        spikeDF = spikeDF[simple_pc_bool] # Remove image cells from the DataFrame

    #%% Population vector analysis # indices of switches in bd or ibd
    raw_mat1 = np.zeros([len(spikeDF), Nbins])
    raw_mat2 = np.zeros([len(spikeDF), Nbins])
    raw_mat_1 = np.zeros([len(spikeDF), Nbins])
    raw_mat_2 = np.zeros([len(spikeDF), Nbins])

    z_mat1_even, z_mat1_odd = np.zeros([len(spikeDF), Nbins]), np.zeros([len(spikeDF), Nbins])
    z_mat2_even, z_mat2_odd = np.zeros([len(spikeDF), Nbins]), np.zeros([len(spikeDF), Nbins])
    z_mat_1_even, z_mat_1_odd = np.zeros([len(spikeDF), Nbins]), np.zeros([len(spikeDF), Nbins])
    z_mat_2_even, z_mat_2_odd = np.zeros([len(spikeDF), Nbins]), np.zeros([len(spikeDF), Nbins])

    for _, ch in enumerate(set(cellResultsDB.choice)):

        for q, cell_id in enumerate(spikeDF.T):
            resultsDB = cellResultsDB[cellResultsDB['cell_id'] == cell_id]

            rt = np.nan_to_num((resultsDB[resultsDB.choice==ch].normRate - resultsDB[resultsDB.choice==ch].normRate.mean()) / resultsDB[resultsDB.choice==ch].normRate.std())
            evenRate = np.nan_to_num((resultsDB[resultsDB.choice==ch].evenRateNorm - resultsDB[resultsDB.choice==ch].evenRateNorm.mean()) / resultsDB[resultsDB.choice==ch].evenRateNorm.std())
            oddRate = np.nan_to_num((resultsDB[resultsDB.choice==ch].oddRateNorm - resultsDB[resultsDB.choice==ch].oddRateNorm.mean()) / resultsDB[resultsDB.choice==ch].oddRateNorm.std())

            if ch==1:
                raw_mat1[q] = rt#resultsDB[resultsDB.choice==ch].normRate / resultsDB.normRate.max()
                z_mat1_even[q] = evenRate
                z_mat1_odd[q] = oddRate
            elif ch==2:
                raw_mat2[q] = rt#resultsDB[resultsDB.choice==ch].normRate / resultsDB.normRate.max()
                z_mat2_even[q] = evenRate
                z_mat2_odd[q] = oddRate
            elif ch==-1:
                raw_mat_1[q] = rt#resultsDB[resultsDB.choice==ch].normRate / resultsDB.normRate.max()
                z_mat_1_odd[q] = oddRate
                z_mat_1_even[q] = evenRate
            elif ch==-2:
                raw_mat_2[q] = rt#resultsDB[resultsDB.choice==ch].normRate / resultsDB.normRate.max()
                z_mat_2_even[q] = evenRate
                z_mat_2_odd[q] = oddRate

    choices = [raw_mat1, raw_mat_1, raw_mat2, raw_mat_2]
    correlations_mat = np.zeros([16,Nbins])
    correlations_shuf = np.zeros([16,Nbins])

    # correlations done for each spatial bin, not each neuron
    for i, (mat1, mat2) in enumerate(it.product(choices, repeat=2)):
        if i == 0:
            correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(z_mat1_even.T, z_mat1_odd.T)]

            shmat1 = z_mat1_even[np.random.permutation(z_mat1_even.shape[0])]
            correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, z_mat1_odd.T)]
        elif i ==5:
            correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(z_mat_1_even.T, z_mat_1_odd.T)]

            shmat1 = z_mat_1_even[np.random.permutation(z_mat_1_even.shape[0])]
            correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, z_mat_1_odd.T)]
        elif i == 10:
            correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(z_mat2_even.T, z_mat2_odd.T)]

            shmat1 = z_mat2_even[np.random.permutation(z_mat2_even.shape[0])]
            correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, z_mat2_odd.T)]
        elif i == 15:
            correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(z_mat_2_even.T, z_mat_2_odd.T)]

            shmat1 = z_mat_2_even[np.random.permutation(z_mat_2_even.shape[0])]
            correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, z_mat_2_odd.T)]

        else:
            correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(mat1.T, mat2.T)]

            shmat1 = mat1[np.random.permutation(mat1.shape[0])]
            correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, mat2.T)]

    correlations_mat_sessions[il] = correlations_mat.copy()
    correlations_shuf_sessions[il] = correlations_shuf.copy()

        #%% Population vector correlation of ITI data
    if doITI == True:
        iti_raw_mat1 = np.zeros([len(spikeDF), 200])
        iti_raw_mat2 = np.zeros([len(spikeDF), 200])
        iti_raw_mat_1 = np.zeros([len(spikeDF), 200])
        iti_raw_mat_2 = np.zeros([len(spikeDF), 200])

        iti_mat1_even, iti_mat1_odd = np.zeros([len(spikeDF), 200]), np.zeros([len(spikeDF), 200])
        iti_mat2_even, iti_mat2_odd = np.zeros([len(spikeDF), 200]), np.zeros([len(spikeDF), 200])
        iti_mat_1_even, iti_mat_1_odd = np.zeros([len(spikeDF), 200]), np.zeros([len(spikeDF), 200])
        iti_mat_2_even, iti_mat_2_odd = np.zeros([len(spikeDF), 200]), np.zeros([len(spikeDF), 200])

        for _, ch in enumerate(set(itiResultsDB.choice)):

            for q, cell_id in enumerate(spikeDF.T):
                itiDB = itiResultsDB[itiResultsDB['cell_id'] == cell_id]

                rt = np.nan_to_num((itiDB[itiDB.choice==ch].normRate - itiDB[itiDB.choice==ch].normRate.mean()) / itiDB[itiDB.choice==ch].normRate.std())
                evenRate = np.nan_to_num((itiDB[itiDB.choice==ch].evenRateNorm - itiDB[itiDB.choice==ch].evenRateNorm.mean()) / itiDB[itiDB.choice==ch].evenRateNorm.std())
                oddRate = np.nan_to_num((itiDB[itiDB.choice==ch].oddRateNorm - itiDB[itiDB.choice==ch].oddRateNorm.mean()) / itiDB[itiDB.choice==ch].oddRateNorm.std())

                if ch==1:
                    iti_raw_mat1[q] = rt#itiDB[itiDB.choice==ch].normRate / itiDB.normRate.max()
                    iti_mat1_even[q] = evenRate
                    iti_mat1_odd[q] = oddRate
                elif ch==2:
                    iti_raw_mat2[q] = rt#itiDB[itiDB.choice==ch].normRate / itiDB.normRate.max()
                    iti_mat2_even[q] = evenRate
                    iti_mat2_odd[q] = oddRate
                elif ch==-1:
                    iti_raw_mat_1[q] = rt#itiDB[itiDB.choice==ch].normRate / itiDB.normRate.max()
                    iti_mat_1_odd[q] = oddRate
                    iti_mat_1_even[q] = evenRate
                elif ch==-2:
                    iti_raw_mat_2[q] = rt#itiDB[itiDB.choice==ch].normRate / itiDB.normRate.max()
                    iti_mat_2_even[q] = evenRate
                    iti_mat_2_odd[q] = oddRate

        iti_choices = [iti_raw_mat1, iti_raw_mat_1, iti_raw_mat2, iti_raw_mat_2]
        iti_correlations_mat = np.zeros([16,200])
        iti_correlations_fl = np.zeros([16,200])
        iti_correlations_shuf = np.zeros([16,200])

        for i, (mat1, mat2) in enumerate(it.product(iti_choices, repeat=2)):
            if i == 0:
                iti_correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(iti_mat1_even.T, iti_mat1_odd.T)]

                shmat1 = iti_mat1_even[np.random.permutation(iti_mat1_even.shape[0])]
                iti_correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, iti_mat1_odd.T)]
            elif i ==5:
                iti_correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(iti_mat_1_even.T, iti_mat_1_odd.T)]

                shmat1 = iti_mat_1_even[np.random.permutation(iti_mat_1_even.shape[0])]
                iti_correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, iti_mat_1_odd.T)]
            elif i == 10:
                iti_correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(iti_mat2_even.T, iti_mat2_odd.T)]

                shmat1 = iti_mat2_even[np.random.permutation(iti_mat2_even.shape[0])]
                iti_correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, iti_mat2_odd.T)]
            elif i == 15:
                iti_correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(iti_mat_2_even.T, iti_mat_2_odd.T)]

                shmat1 = iti_mat_2_even[np.random.permutation(iti_mat_2_even.shape[0])]
                iti_correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, iti_mat_2_odd.T)]
            else:
                iti_correlations_mat[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(mat1.T, mat2.T)]

                shmat1 = mat1[np.random.permutation(mat1.shape[0])] # shuffle neuron labels
                iti_correlations_shuf[i] = [np.corrcoef(pos, neg)[0,1] for pos, neg in zip(shmat1.T, mat2.T)]

        iti_correlations_mat_sessions[il] = iti_correlations_mat.copy()
        iti_correlations_shuf_sessions[il] = iti_correlations_shuf.copy()

#%% pop_vec correlation - all vs shuffled

if not (simSwap or best6 or toExcludeImageCells):
    df_count.to_csv(combinedResultDir+'table_S1_place_cell_counts_pop_vec.csv')

if (best6 or simSwap):
    fig, axx = pl.subplots(1,2,figsize=(3.4,1.4), sharey=True)
else:
    fig, axx = pl.subplots(1,2,figsize=(3.6,1.9), sharey=True)

axx = axx.ravel()

oe_sem1 = np.std(correlations_mat_sessions[:,0],axis=0) / np.sqrt(len(fileList))
oe_mean1 = np.mean(correlations_mat_sessions[:,0],axis=0)

oe_sem_1 = np.std(correlations_mat_sessions[:,5],axis=0) / np.sqrt(len(fileList))
oe_mean_1 = np.mean(correlations_mat_sessions[:,5],axis=0)

oe_sem2 = np.std(correlations_mat_sessions[:,10],axis=0) / np.sqrt(len(fileList))
oe_mean2 = np.mean(correlations_mat_sessions[:,10],axis=0)

oe_sem_2 = np.std(correlations_mat_sessions[:,15],axis=0) / np.sqrt(len(fileList))
oe_mean_2 = np.mean(correlations_mat_sessions[:,15],axis=0)

sem0 = np.std(correlations_mat_sessions[:,0],axis=0) / np.sqrt(len(fileList))
y0 = np.mean(correlations_mat_sessions[:,0],axis=0)
sem5 = np.std(correlations_mat_sessions[:,5],axis=0) / np.sqrt(len(fileList))
y5 = np.mean(correlations_mat_sessions[:,5],axis=0)
sem10 = np.std(correlations_mat_sessions[:,10],axis=0) / np.sqrt(len(fileList))
y10 = np.mean(correlations_mat_sessions[:,10],axis=0)
sem11 = np.std(correlations_mat_sessions[:,11],axis=0) / np.sqrt(len(fileList))
y11 = np.mean(correlations_mat_sessions[:,11],axis=0)
sem15 = np.std(correlations_mat_sessions[:,15],axis=0) / np.sqrt(len(fileList))
y15 = np.mean(correlations_mat_sessions[:,15],axis=0)

sem1 = np.std(correlations_mat_sessions[:,1],axis=0) / np.sqrt(len(fileList))
y1 = np.mean(correlations_mat_sessions[:,1],axis=0)
sem2 = np.std(correlations_mat_sessions[:,2],axis=0) / np.sqrt(len(fileList))
y2 = np.mean(correlations_mat_sessions[:,2],axis=0)
sem3 = np.std(correlations_mat_sessions[:,3],axis=0) / np.sqrt(len(fileList))
y3 = np.mean(correlations_mat_sessions[:,3],axis=0)
sem6 = np.std(correlations_mat_sessions[:,6],axis=0) / np.sqrt(len(fileList))
y6 = np.mean(correlations_mat_sessions[:,6],axis=0)
sem7 = np.std(correlations_mat_sessions[:,7],axis=0) / np.sqrt(len(fileList))
y7 = np.mean(correlations_mat_sessions[:,7],axis=0)

axx[0].plot(xCentersReal,oe_mean1,'r',label='R')
axx[0].fill_between(xCentersReal, oe_mean1-oe_sem1, oe_mean1+oe_sem1, color='r', alpha=0.3)
axx[1].plot(xCentersReal,oe_mean_1,'b')
axx[1].fill_between(xCentersReal, oe_mean_1-oe_sem_1, oe_mean_1+oe_sem_1, color='b', alpha=0.3)
axx[0].plot(xCentersReal,y1,'b',label='L')
axx[0].fill_between(xCentersReal, y1-sem1, y1+sem1, color='b', alpha=0.3)
axx[0].plot(xCentersReal, y2,'m',label='R*')
axx[0].fill_between(xCentersReal, y2-sem2, y2+sem2, color='m', alpha=0.3)
axx[0].plot(xCentersReal, y3,'c',label='L*')
axx[0].fill_between(xCentersReal, y3-sem3, y3+sem3, color='c', alpha=0.3)
axx[0].fill_between([bd[0],bd[1]],-.2,1,facecolor='k',alpha=0.2) # [bd[0],bd[1]]
axx[0].fill_between([bd[2],bd[3]],-.2,1,facecolor='k',alpha=0.2) # [bd[2],bd[3]]
axx[0].set_ylim([0,1])
axx[0].set_xlim([0,totalMazeLength])
axx[0].set_ylabel('Population Vector Correlation')
axx[0].set_xlabel('Track Position (cm)')
axx[0].set_title('R')

axx[1].plot(xCentersReal, y1,'r')
axx[1].fill_between(xCentersReal, y1-sem1, y1+sem1, color='r',alpha=0.3)
axx[1].plot(xCentersReal, y6,'m')
axx[1].fill_between(xCentersReal, y6-sem6, y6+sem6, color='m',alpha=0.3)
axx[1].plot(xCentersReal, y7,'c')
axx[1].fill_between(xCentersReal, y7-sem7, y7+sem7, color='c',alpha=0.3)
axx[1].fill_between([bd[0],bd[1]],-.2,1,facecolor='k',alpha=0.2) # [bd[0],bd[1]]
axx[1].fill_between([bd[2],bd[3]],-.2,1,facecolor='k',alpha=0.2) # [bd[2],bd[3]]
axx[1].set_ylim([0,1])
axx[1].set_xlim([0,totalMazeLength])
axx[0].set_xticks([0,200,400,600])
axx[1].set_xticks([0,200,400,600])
axx[1].set_xlabel('Track Position (cm)')
axx[1].set_title('L')

# shuffled
oe_sem1 = np.std(correlations_shuf_sessions[:,0],axis=0) / np.sqrt(len(fileList))
oe_mean1 = np.mean(correlations_shuf_sessions[:,0],axis=0)

oe_sem_1 = np.std(correlations_shuf_sessions[:,5],axis=0) / np.sqrt(len(fileList))
oe_mean_1 = np.mean(correlations_shuf_sessions[:,5],axis=0)

oe_sem2 = np.std(correlations_shuf_sessions[:,10],axis=0) / np.sqrt(len(fileList))
oe_mean2 = np.mean(correlations_shuf_sessions[:,10],axis=0)

oe_sem_2 = np.std(correlations_shuf_sessions[:,15],axis=0) / np.sqrt(len(fileList))
oe_mean_2 = np.mean(correlations_shuf_sessions[:,15],axis=0)

sem1 = np.std(correlations_shuf_sessions[:,1],axis=0) / np.sqrt(len(fileList))
y1 = np.mean(correlations_shuf_sessions[:,1],axis=0)
sem2 = np.std(correlations_shuf_sessions[:,2],axis=0) / np.sqrt(len(fileList))
y2 = np.mean(correlations_shuf_sessions[:,2],axis=0)
sem3 = np.std(correlations_shuf_sessions[:,3],axis=0) / np.sqrt(len(fileList))
y3 = np.mean(correlations_shuf_sessions[:,3],axis=0)
sem6 = np.std(correlations_shuf_sessions[:,6],axis=0) / np.sqrt(len(fileList))
y6 = np.mean(correlations_shuf_sessions[:,6],axis=0)
sem7 = np.std(correlations_shuf_sessions[:,7],axis=0) / np.sqrt(len(fileList))
y7 = np.mean(correlations_shuf_sessions[:,7],axis=0)

axx[0].plot(xCentersReal,oe_mean1,':r')
axx[0].fill_between(xCentersReal, oe_mean1-oe_sem1, oe_mean1+oe_sem1, color='r', alpha=0.3)
axx[0].plot(xCentersReal,y1,':b')
axx[0].fill_between(xCentersReal, y1-sem1, y1+sem1, color='b', alpha=0.3)
axx[0].plot(xCentersReal, y2,':m')
axx[0].fill_between(xCentersReal, y2-sem2, y2+sem2, color='m', alpha=0.3)
axx[0].plot(xCentersReal, y3,':c')
axx[0].fill_between(xCentersReal, y3-sem3, y3+sem3, color='c', alpha=0.3)
axx[0].set_ylim([-0.2,1])
axx[0].set_xlim([0,totalMazeLength])
axx[0].legend()

axx[1].plot(xCentersReal, y1,':r',label='shuffled R')
axx[1].fill_between(xCentersReal, y1-sem1, y1+sem1, color='r',alpha=0.3)
axx[1].plot(xCentersReal,oe_mean_1,':b',label='shuffled L')
axx[1].fill_between(xCentersReal, oe_mean_1-oe_sem_1, oe_mean_1+oe_sem_1, color='b', alpha=0.3)
axx[1].plot(xCentersReal, y6,':m',label='shuffled R*')
axx[1].fill_between(xCentersReal, y6-sem6, y6+sem6, color='m',alpha=0.3)
axx[1].plot(xCentersReal, y7,':c',label='shuffled L*')
axx[1].fill_between(xCentersReal, y7-sem7, y7+sem7, color='c',alpha=0.3)
axx[1].set_ylim([-0.2,1])
axx[1].set_xlim([0,totalMazeLength])
axx[1].legend()

if toExcludeImageCells:
    pl.savefig(combinedResultDir+'fig_4B_pop_vec_correlation_noImageCells.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)
elif simSwap:
    pl.savefig(combinedResultDir+'fig_S5A_pop_vec_correlation.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)
elif best6:
    pl.savefig(combinedResultDir+'fig_S5C_pop_vec_correlation.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)
else:
    pl.savefig(combinedResultDir+'fig_4A_pop_vec_correlation.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close(fig)

#%% pop_vec correlation - all vs shuffled for each gerbil
if not (toExcludeImageCells or simSwap or best6): # Only do this is using 18 sessions

    indices = np.arange(0,19,3)
    gerbils = ['0395','0397','2017','2018','2783','2784']
    for i, (gid, u, e) in enumerate(zip(gerbils, indices,indices[1:])):
        fig, aq = pl.subplots(1,2,figsize=(3.4,1.2), sharey=True)
        fig.suptitle('Gerbil {}'.format(gid), y=1.1,fontsize=8)
        aq = aq.ravel()

        correlations_mat_sessions_gerbil = correlations_mat_sessions[u:e,:,:]
        correlations_shuf_sessions_gerbil = correlations_shuf_sessions[u:e,:,:]

        oe_sem1 = np.std(correlations_mat_sessions_gerbil[:,0],axis=0) / np.sqrt(e-u)
        oe_mean1 = np.mean(correlations_mat_sessions_gerbil[:,0],axis=0)

        oe_sem_1 = np.std(correlations_mat_sessions_gerbil[:,5],axis=0) / np.sqrt(e-u)
        oe_mean_1 = np.mean(correlations_mat_sessions_gerbil[:,5],axis=0)

        sem1 = np.std(correlations_mat_sessions_gerbil[:,1],axis=0) / np.sqrt(e-u)
        y1 = np.mean(correlations_mat_sessions_gerbil[:,1],axis=0)
        sem2 = np.std(correlations_mat_sessions_gerbil[:,2],axis=0) / np.sqrt(e-u)
        y2 = np.mean(correlations_mat_sessions_gerbil[:,2],axis=0)
        sem3 = np.std(correlations_mat_sessions_gerbil[:,3],axis=0) / np.sqrt(e-u)
        y3 = np.mean(correlations_mat_sessions_gerbil[:,3],axis=0)
        sem6 = np.std(correlations_mat_sessions_gerbil[:,6],axis=0) / np.sqrt(e-u)
        y6 = np.mean(correlations_mat_sessions_gerbil[:,6],axis=0)
        sem7 = np.std(correlations_mat_sessions_gerbil[:,7],axis=0) / np.sqrt(e-u)
        y7 = np.mean(correlations_mat_sessions_gerbil[:,7],axis=0)

        aq[0].plot(xCentersReal,oe_mean1,'r',label='R')
        aq[0].fill_between(xCentersReal, oe_mean1-oe_sem1, oe_mean1+oe_sem1, color='r', alpha=0.3)
        aq[1].plot(xCentersReal,oe_mean_1,'b')
        aq[1].fill_between(xCentersReal, oe_mean_1-oe_sem_1, oe_mean_1+oe_sem_1, color='b', alpha=0.3)

        aq[0].plot(xCentersReal,y1,'b',label='L')
        aq[0].fill_between(xCentersReal, y1-sem1, y1+sem1, color='b', alpha=0.3)
        aq[0].plot(xCentersReal, y2,'m',label='R*')
        aq[0].fill_between(xCentersReal, y2-sem2, y2+sem2, color='m', alpha=0.3)
        aq[0].plot(xCentersReal, y3,'c',label='L*')
        aq[0].fill_between(xCentersReal, y3-sem3, y3+sem3, color='c', alpha=0.3)
        aq[0].fill_between([bd[0],bd[1]],-.2,1,facecolor='k',alpha=0.2) # [bd[0],bd[1]]
        aq[0].fill_between([bd[2],bd[3]],-.2,1,facecolor='k',alpha=0.2) # [bd[2],bd[3]]
        aq[0].set_ylim([-0.2,1])
        aq[0].set_xlim([0,totalMazeLength])
        aq[0].set_ylabel('Population Vector Correlation')
        aq[0].set_xlabel('Track Position (cm)')
        aq[0].set_title('R')
        aq[0].set_yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])

        aq[1].plot(xCentersReal, y1,'r')
        aq[1].fill_between(xCentersReal, y1-sem1, y1+sem1, color='r',alpha=0.3)
        aq[1].plot(xCentersReal, y6,'m')
        aq[1].fill_between(xCentersReal, y6-sem6, y6+sem6, color='m',alpha=0.3)
        aq[1].plot(xCentersReal, y7,'c')
        aq[1].fill_between(xCentersReal, y7-sem7, y7+sem7, color='c',alpha=0.3)
        aq[1].fill_between([bd[0],bd[1]],-.2,1,facecolor='k',alpha=0.2) # [bd[0],bd[1]]
        aq[1].fill_between([bd[2],bd[3]],-.2,1,facecolor='k',alpha=0.2) # [bd[2],bd[3]]
        aq[1].set_ylim([-0.2,1])
        aq[1].set_xlim([0,totalMazeLength])
        aq[0].set_xticks([0,200,400,600])
        aq[1].set_xticks([0,200,400,600])
        aq[1].set_xlabel('Track Position (cm)')
        aq[1].set_title('L')
        aq[1].set_yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])

        pl.savefig(combinedResultDir+'fig_S5E_pop_vec_correlation_g{}.pdf'.format(gid),format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        pl.close(fig)

#%% Plot the matrix for the poulation vector correlation divided by maze segment
if not (toExcludeImageCells or simSwap or best6):

    parts = [':',':23', '23:34', '34:53', '53:63', '63:']
    fig, axs = pl.subplots(3,2, figsize=(3.2,4.6),sharex=True, sharey=True)
    axs = axs.ravel()

    allavgcorrelations_mat = np.mean(correlations_mat_sessions,axis=0)

    for i in range(6):
        if i == 0:
            temp_mat = allavgcorrelations_mat.copy()
            axs[i].set_title('Entire Mazes',pad=2)
        elif i == 1:
            temp_mat = allavgcorrelations_mat[:,:24].copy()
            axs[i].set_title('First Hallway',pad=2)
        elif i == 2:
            temp_mat = allavgcorrelations_mat[:,23:34].copy()
            axs[i].set_title('First Corner',pad=2)
        elif i == 3:
            temp_mat = allavgcorrelations_mat[:,34:54].copy()
            axs[i].set_title('Middle Hallway',pad=2)
        elif i == 4:
            temp_mat = allavgcorrelations_mat[:,53:64].copy()
            axs[i].set_title('Second Corner',pad=2)
        elif i == 5:
            temp_mat = allavgcorrelations_mat[:,63:].copy()
            axs[i].set_title('Final Hallway',pad=2)

        temp_mat = temp_mat.T # Transpose matrix to accord for proper dimensions after removing 3rd dim orignially containing multiple sessions.
        k =[
                [np.mean(temp_mat[:,0]), np.mean(temp_mat[:,1]),
                 np.mean(temp_mat[:,2]), np.mean(temp_mat[:,3])],
                [np.mean(temp_mat[:,4]), np.mean(temp_mat[:,5]),
                 np.mean(temp_mat[:,6]), np.mean(temp_mat[:,7])],
                [np.mean(temp_mat[:,8]), np.mean(temp_mat[:,9]),
                 np.mean(temp_mat[:,10]), np.mean(temp_mat[:,11])],
                [np.mean(temp_mat[:,12]), np.mean(temp_mat[:,13]),
                 np.mean(temp_mat[:,14]), np.mean(temp_mat[:,15])] ]
        im = axs[i].imshow(k, 'Oranges', vmin=0, vmax=0.6) # need to change the color bar ticks below

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.125, 0.015, 0.755])
        fig.colorbar(im, cax=cbar_ax, ticks=[0,0.6]) # must match the vmin and vmax above
        cbar_ax.tick_params(labelsize=6)
        cbar_ax.set_ylabel("r",rotation=0,labelpad=-13)
        axs[i].set_xticklabels([0, 1, 2, 3])
        axs[i].set_yticklabels([0, 1, 2, 3])

        axs[i].set(
            xticks = [0, 1, 2, 3],
            xticklabels=(mazeTypeList),
            yticks = [0, 1, 2, 3],
            yticklabels=(mazeTypeList)
            );

        axs[i].grid(False)

    pl.savefig(combinedResultDir+'fig_4C_population_vector_correlation_bySegment_v.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close()

#%% Plot the matrix for the poulation vector correlation divided by maze segment
if (simSwap or best6):
    parts = [':',':23', '23:34', '34:53', '53:63', '63:']
    fig, axw = pl.subplots(2,3, figsize=(3.4,2.2),sharex=True, sharey=True)
    axw = axw.ravel()

    allavgcorrelations_mat = np.mean(correlations_mat_sessions,axis=0)

    for i in range(6):
        if i == 0:
            temp_mat = allavgcorrelations_mat.copy()
            axw[i].set_title('Entire Mazes',pad=2)
        elif i == 1:
            temp_mat = allavgcorrelations_mat[:,:24].copy()
            axw[i].set_title('First Hallway', pad=2)
        elif i == 2:
            temp_mat = allavgcorrelations_mat[:,23:34].copy()
            axw[i].set_title('First Corner',pad=2)
        elif i == 3:
            temp_mat = allavgcorrelations_mat[:,34:54].copy()
            axw[i].set_title('Middle Hallway',pad=2)
        elif i == 4:
            temp_mat = allavgcorrelations_mat[:,53:64].copy()
            axw[i].set_title('Second Corner', pad=2)
        elif i == 5:
            temp_mat = allavgcorrelations_mat[:,63:].copy()
            axw[i].set_title('Final Hallway',pad=2)

        temp_mat = temp_mat.T # Transpose matrix to accord for proper dimensions after removing 3rd dim orignially containing multiple sessions.
        k =[
                [np.mean(temp_mat[:,0]), np.mean(temp_mat[:,1]),
                 np.mean(temp_mat[:,2]), np.mean(temp_mat[:,3])],
                [np.mean(temp_mat[:,4]), np.mean(temp_mat[:,5]),
                 np.mean(temp_mat[:,6]), np.mean(temp_mat[:,7])],
                [np.mean(temp_mat[:,8]), np.mean(temp_mat[:,9]),
                 np.mean(temp_mat[:,10]), np.mean(temp_mat[:,11])],
                [np.mean(temp_mat[:,12]), np.mean(temp_mat[:,13]),
                 np.mean(temp_mat[:,14]), np.mean(temp_mat[:,15])] ]
        im = axw[i].imshow(k, 'Oranges', vmin=0, vmax=0.6) # need to change the color bar ticks below

        fig.subplots_adjust(right=0.9)
        cbar_x = fig.add_axes([0.92, 0.125, 0.015, 0.755])
        fig.colorbar(im, cax=cbar_x, ticks=[0,0.6]) # must match the vmin and vmax above
        cbar_x.tick_params(labelsize=6)
        cbar_x.set_ylabel("r",rotation=0,labelpad=-13)
        axw[i].set_xticklabels([0, 1, 2, 3])
        axw[i].set_yticklabels([0, 1, 2, 3])

        axw[i].set(
            xticks = [0, 1, 2, 3],
            xticklabels=(mazeTypeList),
            yticks = [0, 1, 2, 3],
            yticklabels=(mazeTypeList)
            );

        axw[i].grid(False)

    if simSwap:
        pl.savefig(combinedResultDir+'fig_S5B_population_vector_correlation_bySegment_simSwap.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        pl.close()
    if best6:
        pl.savefig(combinedResultDir+'fig_S5D_population_vector_correlation_bySegment_best6.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        pl.close()

#%% Plot grouped results for iti Analysis
# Line plot showing the correlations
if not (toExcludeImageCells or simSwap or best6) and doITI:

    fig, ax = pl.subplots(1,4, figsize=(4.88,1.1),sharex=True, sharey=True)
    ax = ax.ravel()

    oe_sem1 = np.std(iti_correlations_mat_sessions[:,0],axis=0) / np.sqrt(len(fileList))
    oe_mean1 = np.mean(iti_correlations_mat_sessions[:,0],axis=0)

    oe_sem_1 = np.std(iti_correlations_mat_sessions[:,5],axis=0) / np.sqrt(len(fileList))
    oe_mean_1 = np.mean(iti_correlations_mat_sessions[:,5],axis=0)

    oe_sem2 = np.std(iti_correlations_mat_sessions[:,10],axis=0) / np.sqrt(len(fileList))
    oe_mean2 = np.mean(iti_correlations_mat_sessions[:,10],axis=0)

    oe_sem_2 = np.std(iti_correlations_mat_sessions[:,15],axis=0) / np.sqrt(len(fileList))
    oe_mean_2 = np.mean(iti_correlations_mat_sessions[:,15],axis=0)

    sem11 = np.std(iti_correlations_mat_sessions[:,11],axis=0) / np.sqrt(len(fileList))
    y11 = np.mean(iti_correlations_mat_sessions[:,11],axis=0)

    sem1 = np.std(iti_correlations_mat_sessions[:,1],axis=0) / np.sqrt(len(fileList))
    y1 = np.mean(iti_correlations_mat_sessions[:,1],axis=0)
    sem2 = np.std(iti_correlations_mat_sessions[:,2],axis=0) / np.sqrt(len(fileList))
    y2 = np.mean(iti_correlations_mat_sessions[:,2],axis=0)
    sem3 = np.std(iti_correlations_mat_sessions[:,3],axis=0) / np.sqrt(len(fileList))
    y3 = np.mean(iti_correlations_mat_sessions[:,3],axis=0)
    sem6 = np.std(iti_correlations_mat_sessions[:,6],axis=0) / np.sqrt(len(fileList))
    y6 = np.mean(iti_correlations_mat_sessions[:,6],axis=0)
    sem7 = np.std(iti_correlations_mat_sessions[:,7],axis=0) / np.sqrt(len(fileList))
    y7 = np.mean(iti_correlations_mat_sessions[:,7],axis=0)

    shy1 = np.mean(iti_correlations_shuf_sessions[:,1],axis=0)
    shy2 = np.mean(iti_correlations_shuf_sessions[:,2],axis=0)
    shy3 = np.mean(iti_correlations_shuf_sessions[:,3],axis=0)
    shy6 = np.mean(iti_correlations_shuf_sessions[:,6],axis=0)
    shy7 = np.mean(iti_correlations_shuf_sessions[:,7],axis=0)
    shy4 = np.mean(iti_correlations_shuf_sessions[:,4],axis=0)
    shy5 = np.mean(iti_correlations_shuf_sessions[:,5],axis=0)
    shy8 = np.mean(iti_correlations_shuf_sessions[:,8],axis=0)
    shy9 = np.mean(iti_correlations_shuf_sessions[:,9],axis=0)
    shy10 = np.mean(iti_correlations_shuf_sessions[:,10],axis=0)
    shy11 = np.mean(iti_correlations_shuf_sessions[:,11],axis=0)
    shy12 = np.mean(iti_correlations_shuf_sessions[:,12],axis=0)
    shy13 = np.mean(iti_correlations_shuf_sessions[:,13],axis=0)
    shy14 = np.mean(iti_correlations_shuf_sessions[:,14],axis=0)
    shy15 = np.mean(iti_correlations_shuf_sessions[:,15],axis=0)

    timeAxis = np.unique(itiResultsDB.itiTimes)
    ax[0].plot(timeAxis,oe_mean1,'r')
    ax[0].fill_between(timeAxis, oe_mean1-oe_sem1, oe_mean1+oe_sem1, color='r', alpha=0.3)
    ax[1].plot(timeAxis,oe_mean_1,'b')
    ax[1].fill_between(timeAxis, oe_mean_1-oe_sem_1, oe_mean_1+oe_sem_1, color='b', alpha=0.3)
    ax[2].plot(timeAxis,oe_mean2,'m')
    ax[2].fill_between(timeAxis, oe_mean2-oe_sem2, oe_mean2+oe_sem2, color='m', alpha=0.3)
    ax[3].plot(timeAxis,oe_mean_2,'c')
    ax[3].fill_between(timeAxis, oe_mean_2-oe_sem_2, oe_mean_2+oe_sem_2, color='c', alpha=0.3)

    ax[0].plot(timeAxis, y1,'b')
    ax[0].fill_between(timeAxis, y1-sem1, y1+sem1, color='b',alpha=0.3)
    ax[0].plot(timeAxis, y2,'m')
    ax[0].fill_between(timeAxis, y2-sem2, y2+sem2, color='m',alpha=0.3)
    ax[0].plot(timeAxis, y3,'c')
    ax[0].fill_between(timeAxis, y3-sem3, y3+sem3, color='c',alpha=0.3)
    ax[0].plot(timeAxis,np.array([shy1,shy2,shy3]).max(axis=0),'k')
    ax[0].fill_between(timeAxis, np.array([shy1,shy2,shy3]).mean(axis=0), np.array([shy1,shy2,shy3]).max(axis=0), color='k', alpha=0.3)
    ax[0].set_title('R',pad=2)
    ax[0].legend(['R','L','R*','L*','sh'], frameon=False)
    ax[0].set_ylabel('Population Vector Correlation')

    ax[1].plot(timeAxis, y1,'r')
    ax[1].fill_between(timeAxis, y1-sem1, y1+sem1, color='r',alpha=0.3)
    ax[1].plot(timeAxis, y6,'m')
    ax[1].fill_between(timeAxis, y6-sem6, y6+sem6, color='m',alpha=0.3)
    ax[1].plot(timeAxis, y7,'c')
    ax[1].fill_between(timeAxis, y7-sem7, y7+sem7, color='c',alpha=0.3)

    ax[1].plot(timeAxis,np.array([shy1,shy7,shy6]).max(axis=0),'k')
    ax[1].fill_between(timeAxis, np.array([shy1,shy7,shy6]).mean(axis=0), np.array([shy1,shy7,shy6]).max(axis=0), color='k', alpha=0.3)
    ax[1].set_title('L',pad=2)

    ax[2].plot(timeAxis,y11,'c')
    ax[2].fill_between(timeAxis, y11-sem11, y11+sem11, color='c', alpha=0.3)
    ax[2].plot(timeAxis, y2,'r')
    ax[2].fill_between(timeAxis, y2-sem2, y2+sem2, color='r', alpha=0.3)
    ax[2].plot(timeAxis, y6,'b')
    ax[2].fill_between(timeAxis, y6-sem6, y6+sem6, color='b', alpha=0.3)

    ax[2].plot(timeAxis,np.array([shy11,shy2,shy6]).max(axis=0),'k')
    ax[2].fill_between(timeAxis, np.array([shy11,shy2,shy6]).mean(axis=0), np.array([shy11,shy2,shy6]).max(axis=0), color='k', alpha=0.3)
    ax[2].set_title('R*',pad=2)
    ax[3].plot(timeAxis,y11,'m')
    ax[3].fill_between(timeAxis, y11-sem11, y11+sem11, color='m', alpha=0.3)
    ax[3].plot(timeAxis, y7,'b')
    ax[3].fill_between(timeAxis, y7-sem7, y7+sem7, color='b', alpha=0.3)
    ax[3].plot(timeAxis, y3,'r')
    ax[3].fill_between(timeAxis, y3-sem3, y3+sem3, color='r', alpha=0.3)

    ax[3].plot(timeAxis,np.array([shy11,shy7,shy3]).max(axis=0),'k')
    ax[3].fill_between(timeAxis, np.array([shy11,shy7,shy3]).mean(axis=0), np.array([shy11,shy7,shy3]).max(axis=0), color='k', alpha=0.3)
    ax[3].set_title('L*',pad=2)

    for i in range(4):
        ax[i].set_xticks([0,10,20])
        ax[i].set_xticks([5,15], minor=True)
        ax[i].set_ylim([-0.1,0.6])
        ax[i].set_xlim([0,20])
        ax[i].fill_between([2,2.1],-.2,1,facecolor='k',alpha=0.5)
        ax[i].fill_between([12,12.1],-.2,1,facecolor='k',alpha=0.5)
        ax[i].set_xlabel('ITI Time (Seconds)')

    pl.savefig(combinedResultDir+'fig_5A_iti_pop_vec_correlation.pdf',format='pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
    pl.close()
