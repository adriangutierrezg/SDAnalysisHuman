#!/usr/bin/env python3
#adriangutierrezg 

"""Loading an .ncs Pegasus file

adriangutierrezg@gmail.com
"""

import os
import numpy as np
import sys
from scipy.signal import spectrogram, butter, lfilter
import pandas as pd
from szs_path import SZSPATH
#szp_path = '/media/Projects/Adrian/seizureProcessing/'
sys.path.insert(0, SZSPATH)
from seizureProcessing.prepare_directory.prepare_dir import *
from seizureProcessing.utils.get_chnames import *
from seizureProcessing.utils.get_patinfo_from_path import *
from seizureProcessing.plotter.plot_raster import *
#from seizureProcessing.utils.read_spikedata import extract_from_combinato
from seizureProcessing.utils.from_micro_to_macro_names import *
from seizureProcessing.utils.readMatfile import *
from seizureProcessing.utils.readSpikeData import *
from seizureProcessing.utils.loadNCS import *


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_psd_per_bands(path,seizure_offset_sec, sd_list, sf=1000, nperseg=1000, min_to_try=25, period_sec=60,
                         min_for_linear_regression=15, mode='median', so_sec=30*60):
    rerefpath = find_dir(path, 'reref')
    fnames = match_cssfiles_to_chnames(path)
    pnr, sznr = get_patinfo_from_path(path)
    chnames =get_chnames(path, strip=True)
    matfiles_data = [x.strip('.ncs')+'_reref_data.mat' for x in fnames]
    matfiles_times = [x.strip('.ncs')+'_reref_times.mat' for x in fnames]
    
    #stating constants
    band_names = ['sub_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    band_lows = [0, 1, 4, 8, 14, 30, 80]
    band_ups = [1, 4, 8, 13, 30, 80, 150]
    metrics = ['baseline_mean','preictal_mean','posictal_mean', 'posictal_diff_norm',
               'A', 'B', 'time_to_preictal_recovery']
    so_index = so_sec*sf
    
    
    #split 
    bl_indices = np.array_split(np.arange(0,60*20*sf), 20)
    bl_idx_choices = np.random.choice(np.arange(len(bl_indices)), size=10, replace=True)
    
    so_corrected = so_index-10
    preictal_indices = np.arange(so_corrected-(60*sf), (so_corrected))
    
    posictal_index = (seizure_offset_sec*sf)#+(10*sf) #avoiding SD
    posictal_indices = np.array_split(np.arange(posictal_index, posictal_index+(min_to_try*60*sf)), min_to_try)
    
    #Cut data by frequency band
    
    #creating empty dic for output
    out_dic = dict()
    bands = pd.DataFrame()
    df = pd.DataFrame()
    
    for m in metrics:
        out_dic[m] = dict()
        for i, ch in enumerate(matfiles_data):
            curr_ch = chnames[i]
            out_dic[m][curr_ch] = []
    
    for i, ch in enumerate(matfiles_data):
        curr_ch = chnames[i]
        print(f'working on {curr_ch}')
        #print(f'extracting from {ch}')
        
        file = readMatfile(os.path.join(rerefpath, ch))
        file_ts = readMatfile(os.path.join(rerefpath, matfiles_times[i]))
        data = file.reref_data
        timestamps = file_ts.reref_times
        
        
        #calculate baseline
        mean_bl = []
        for i, curr_min in enumerate(bl_idx_choices):
            curr_data = data[bl_indices[curr_min]] #cut_data
            
            #get spectogram
            f_axis, t_axis, spg = spectrogram(curr_data, fs=sf, nperseg=1000, noverlap=0, scaling='density')
            #dataframe = pd.DataFrame(spg, columns=(t_axis-t_axis[0])*sf)
            if mode == 'median':
                mean_bl.append(np.median(spg, axis=1))
            else:
                mean_bl.append(np.mean(spg, axis=1))
        
        #for i, val in enumerate(band_lows):
        #    bands[band_names[i]] = np.meadian([val:band_ups[i]])
        
        if mode=='median':
            #out_dic['baseline_mean'][curr_ch].append(np.median(mean_bl))
            out_dic['baseline_mean'][curr_ch] = np.median(mean_bl, axis=0)
        else:
            #out_dic['baseline_mean'][curr_ch].append(np.mean(mean_bl))
            out_dic['baseline_mean'][curr_ch].append(np.mean(mean_bl, axis=0))
        
        #calculate pre-ictal
        curr_data = data[preictal_indices] #cut_data
        #get spectogram
        f_axis, t_axis, spg = spectrogram(curr_data, fs=sf, nperseg=1000, noverlap=0, scaling='density')
        #save data
        if mode=='median':
            #out_dic['preictal_mean'][curr_ch] = np.median(np.median(spg, axis=1))
            out_dic['preictal_mean'][curr_ch] = np.median(spg, axis=1)
        else:
            #out_dic['preictal_mean'][curr_ch] = np.mean(np.mean(spg, axis=1))
            out_dic['preictal_mean'][curr_ch] = np.mean(spg, axis=1)
        
        #calculate all post-ictal
        posictal_psds = []
        for i, curr_pos in enumerate(posictal_indices):
            curr_data = data[curr_pos]
            
            #get spectogram
            f_axis, t_axis, spg = spectrogram(curr_data, fs=sf, nperseg=1000, noverlap=0, scaling='density')
            
            if mode=='median':
                #posictal_psds.append(np.median(np.median(spg, axis=1)))
                posictal_psds.append(np.median(spg, axis=1))
            else:
                #posictal_psds.append(np.mean(np.mean(spg, axis=1)))
                posictal_psds.append(np.mean(spg, axis=1))

        #normalize to baseline
        out_dic['preictal_mean'][curr_ch] = np.divide(out_dic['preictal_mean'][curr_ch],
                                              out_dic['baseline_mean'][curr_ch])
        norm_posictal_psds = [np.divide(x, out_dic['baseline_mean'][curr_ch]) for x in posictal_psds]
        
        
        #calculate linear regression
        out_dic['A'][curr_ch], out_dic['B'][curr_ch] = np.polyfit(np.arange(1, len(posictal_indices)+1), 
                                                                  norm_posictal_psds, 1) 
        time_to_preictal_recovery = (out_dic['preictal_mean'][curr_ch]- out_dic['B'][curr_ch] )/out_dic['A'][curr_ch] 
        
        
        out_dic['time_to_preictal_recovery'][curr_ch] = time_to_preictal_recovery
        out_dic['posictal_mean'][curr_ch] = norm_posictal_psds[0]
    
        out_dic['posictal_diff_norm'][curr_ch] = out_dic['posictal_mean'][curr_ch] - out_dic['preictal_mean'][curr_ch]
        #out_dic[ext_names] = out_dic[ext_names].div(out_dic['baseline_mean'], axis=0) #normalize postictal
        #out_dic['posictal_diff_norm'] = out_dic['1_posictal_mean'] - out_dic['preictal_mean'] #diff normalized post-pre
    
    
    #get indices of 
    indices_bands = []
    for l, u in zip(band_lows, band_ups):
        indices_bands.append((np.where(f_axis==float(l))[0][0], np.where(f_axis==float(u))[0][0]))
    
    dataframes = dict()
    for metric_key, chan in out_dic.items():
        df = pd.DataFrame()
        
        #add all other important data
        df['patID'] = [pnr]*len(chnames)
        df['seizure'] = [sznr]*len(chnames)
        df['chname'] = chnames
        df['region'] = [x[:-1] for x in chnames]
        df['hemisphere'] = [x[0] for x in chnames]
        df['sd'] = df['region'].isin(sd_list)

        out_chnames = []
        for k, v in chan.items():
            for frange, bname in zip(indices_bands, band_names): 
                row_index = df.index[df['chname']==k].tolist()[0]
                if mode=='median':
                    df.loc[row_index, bname] = np.median(v[frange[0]:frange[1]])
                else:
                    df.loc[row_index, bname] = np.mean(v[frange[0]:frange[1]])
        
        dataframes[metric_key] = df
        
        
    return dataframes, metrics

if __name__ == '__main__':
    extract_psd_per_bands()