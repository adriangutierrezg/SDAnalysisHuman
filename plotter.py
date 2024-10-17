#!/usr/bin/env python3
#adriangutierrezg 

'''
'''

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import numpy as np
import os
import sys 
from scipy.signal import spectrogram
from filesystem_handlers import *
from spectral_analysis import *
from szs_path import SZSPATH
from sd_calculation import *
sys.path.insert(0, SZSPATH) #local imports
from seizureProcessing.utils.get_directories import *
from seizureProcessing.utils.get_chnames import *
from seizureProcessing.utils.get_patinfo_from_path import *
from seizureProcessing.utils.readMatfile import *

def plotObject(nrChannels, myDpi=92, lenXax=1900, lenYax=1024, shareX=False, shareY=False):
    '''
    '''
    fig, axs = plt.subplots(nrChannels*2, 1, figsize=(lenXax/myDpi, lenYax/myDpi), sharex=shareX, sharey=shareY)
    gs = gridspec.GridSpec(nrChannels*2, 1, height_ratios=[1, 0.5]*(nrChannels))
    spines = ('top', 'right', 'bottom')
    for i in range(nrChannels*2):
        axs[i] = plt.subplot(gs[i])
        for sp in spines: 
            axs[i].spines[sp].set_visible(False)
        axs[i].tick_params(bottom=False, labelbottom=False)
    axs[i].spines[sp].set_visible(True)
    axs[i].tick_params(bottom=True, labelbottom=True)
    #remove spines
   
    fig.tight_layout()
    fig.subplots_adjust(hspace=.0)
    #fig.tight_layout()
    #plt.show()
    return fig, axs


def plot_determine_onset_regions(path, so_sec=30*60, sf=1000, sec_before=60, sec_after=120, fmin=0, fmax=30, 
                                 plot_filt=False, region_list=None, figlabel=None, save=False, savepath=None,
                                 duration_sec=False, spec_vmin=None, spec_vmax=None):
    
    
    reref_path = adjustPathToLatestPreprocessingStep(path)
    _regions = get_region_name(path)
    
    if region_list:
        regions = {k:v for k,v in _regions.items() if k in region_list}
    else:
        regions = _regions
    del _regions
    
    _ = get_matfile_chnames(reref_path)
    chnames, matfiles = _[0], _[1]
    
    if not savepath:
        savepath = path
    
    #get matfiles grouped by region
    if 'macro' in path and 'reref' in reref_path:
        chnames_reref = [c for c in chnames if '8' not in c]
        matfiles_dic = {key : [matfiles[i] for i, ch in enumerate(chnames_reref) if ch in val] for key, val in regions.items()}
    else:
        matfiles_dic = {key : [matfiles[i] for i, ch in enumerate(chnames) if ch in val] 
                            for key, val in regions.items()}
    #print(matfiles_dic)
    #PLOT PARAMS
    my_dpi = 92
    if not spec_vmin:
        spec_vmin = -fmax
    if not spec_vmax:
        spec_vmax = fmax
    
    # PLOT VARIABLES
    all_regions = list(matfiles_dic.keys())

    if 'macro' in path:

        if os.path.basename(path) == '/':
            session_path = os.path.dirname(os.path.dirname(path))
        else:
            session_path = os.path.dirname(path)
        
        #print(session_path)
        pnr, sznr = get_patinfo_from_path(session_path)
    else:
        pnr, sznr = get_patinfo_from_path(path)
    
    
    if plot_filt:
        if fmin ==0:
            lowfreq = 0.2
        else:
            lowfreq = fmin
        
    for ix, reg in enumerate(all_regions):
        curr_reg = matfiles_dic[reg]
        fig, ax = plotObject(nrChannels=len(curr_reg), lenXax=1000, lenYax=1024, shareY=True)
        offsets = np.linspace(-1000, 1000, len(curr_reg))
        #print(reg)
        for i, ch in enumerate(curr_reg):

            #Fill dataframe info
            #dataDf['pnr'] = 
            file = readMatfile(os.path.join(reref_path, ch))
            data = file.reref_data[(so_sec * sf) - (sec_before * sf):(so_sec * sf) + (sec_after * sf)]

            timeax = np.linspace(-sec_before, sec_after, len(data))
            offset = offsets[i]
            f_axis, t_axis, spg = spectrogram(data, fs=sf, nperseg=200, noverlap=150, scaling='density')


            if plot_filt:
                ax[i*2].plot(timeax, data, c="b")
                filt = butter_bandpass_filter(data, lowfreq, fmax, fs=sf, order=1)
                ax[i*2].plot(timeax, filt, c="red", linewidth=0.5)
            else:
                ax[i*2].plot(timeax, data, c="b", linewidth=0.5)

            ax[i*2].set_ylabel(f'ch{i+1}', fontweight='bold', fontsize=15)
            data_to_plot = 20*np.log10(spg[fmin:fmax,])
            #print(type(data_to_plot))
            spec = ax[(i*2)+1].imshow(data_to_plot, aspect='auto', cmap='hot', origin='lower',
                                     extent=[min(timeax), max(timeax), fmin, fmax],
                                     norm=Normalize(vmin=spec_vmin, vmax=spec_vmax, clip=False))#, vmin=min([min(x) for x in data_to_plot]), vmax=max([min(x) for x in data_to_plot]))

            fig.colorbar(spec, ax=ax[(i*2)+1], orientation='horizontal', pad=0.05)

            ax[i*2].set_xlim(-sec_before, sec_after)
            ax[i*2].set_xticks(np.arange(-sec_before, sec_after, 1), minor=False)
            #ax[i*2].set_xticks(np.arange(-sec_before, sec_after, 0.1), minor=True)

            # Apply grid settings to individual axes
            #ax[i*2].grid(visible=True, which='minor', axis='x')
            ax[i*2].grid(visible=True, which='major', axis='x', linewidth=0.1, color='k')
            
            if duration_sec:
                ax[i*2].vlines(x=[0,duration_sec], ymin=min(data), ymax=max(data), colors='red', linewidths=1)
            else:
                ax[i*2].vlines(x=0, ymin=min(data), ymax=max(data), colors='red', linewidths=1)
            fig.suptitle(f'{reg} (pat.{pnr} szr{sznr})',fontweight="bold", fontsize=16)

        ax[i*2+1].tick_params(bottom=True, labelbottom=True)
        ax[i*2+1].set_xlabel('time (sec)', fontweight='bold', fontsize=15)

        if save:
            spath = os.path.join(savepath, 'plots_so_determination')

            if not os.path.exists(spath):
                print(f'creating path for saving: {spath}')
                os.makedirs(spath)

            figname= f'{reg}_so_determiation'
            
            if figlabel:
                figname += figlabel
            
            plt.savefig(os.path.join(spath, figname+'.png'), bbox_inches='tight')
            print(f'saving plot {figname} in {spath}')
        #figname = label+''
        plt.show()

def plot_sd_filtered_region(path, sd_onset_sec, sf=1000, sec_after=10, adjust_ms=150, order=2, cutoff=2, 
                            reg_list=None, figlabel=None, save=False, savepath=None, duration_sec=None):
    '''
    '''
    
    reref_path = adjustPathToLatestPreprocessingStep(path)
    _regions = get_region_name(path)
    
    if reg_list:
        regions = {k:v for k,v in _regions.items() if k in reg_list}
    else:
        regions = _regions
    del _regions
    
    _ = get_matfile_chnames(reref_path)
    chnames, matfiles = _[0], _[1]
    
    if not savepath:
        savepath = path
    
    #get matfiles grouped by region
    if 'macro' in path and 'reref' in reref_path:
        chnames_reref = [c for c in chnames if '8' not in c]
        matfiles_dic = {key : [matfiles[i] for i, ch in enumerate(chnames_reref) if ch in val] for key, val in regions.items()}
    else:
        matfiles_dic = {key : [matfiles[i] for i, ch in enumerate(chnames) if ch in val] 
                            for key, val in regions.items()}
    
    all_regions = list(matfiles_dic.keys())

    if 'macro' in path:

        if os.path.basename(path) == '/':
            session_path = os.path.dirname(os.path.dirname(path))
        else:
            session_path = os.path.dirname(path)
        
        #print(session_path)
        pnr, sznr = get_patinfo_from_path(session_path)
    else:
        pnr, sznr = get_patinfo_from_path(path)
    
    ch_with_sd = {}
    for ix, reg in enumerate(all_regions):
        curr_reg = matfiles_dic[reg]
        data_in_reg = []
        #chnames_in_reg = []
        #get data
        for _, ch in enumerate(curr_reg):
            file = readMatfile(os.path.join(reref_path, ch))
            data_in_reg.append(file.reref_data[(sd_onset_sec*sf):(sd_onset_sec*sf)+(sec_after*sf)])

        #calculate delays
        wave_delays, sdchans_ix = get_wave_delays_per_wire(data_in_reg, 
                                               onset_adjust_ms=adjust_ms, 
                                               filter_order=order, 
                                               lpf_cutoff_fr_hz=cutoff)
        ch_with_sd[reg] = [curr_reg[i] for i in sdchans_ix]
        #sort indices
        indices = np.argsort(wave_delays)
        indices = indices[:np.count_nonzero(~np.isnan(wave_delays))]
        #print(indices)

        #plot results
        tabcols = list(mcolors.TABLEAU_COLORS.keys())
        plt.figure(figsize=(10,6))

        for i, ind in enumerate(data_in_reg):
            curr_data = ind
            res = butter_lowpass_filter(curr_data, cutoff, sf, order=order)
            curr_data = res[:sec_after*sf]
            plt.plot(curr_data, c='gray')


        for i, ind in enumerate(indices[:]):
            color = tabcols[i+1]

            curr_data = data_in_reg[ind]#[:2000]
            res = butter_lowpass_filter(curr_data, cutoff, 1000, order=order)
            curr_data = res[:sec_after*sf]
            xdot = int(wave_delays[ind])
            plt.plot(curr_data, c=mcolors.TABLEAU_COLORS[tabcols[i]] )
            plt.plot(xdot, curr_data[xdot], marker='o',label=str(xdot)+' ms')
            plt.legend()
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (uV)')

        plt.title(reg+ ' low-pass filtered')
        if duration_sec:
            plt.axvline(int(duration_sec*sf), color='red', linewidth=1)
        if save:
            spath = os.path.join(savepath, 'plots_sd_regions')

            if not os.path.exists(spath):
                print(f'creating path for saving: {spath}')
                os.mkdir(spath)

            figname= f'{reg}_sd_determiation'

            if figlabel:
                figname += figlabel

            plt.savefig(os.path.join(spath, figname+'.png'))
            print(f'saving plot {figname} in {spath}')
        #plt.savefig(os.path.join(out_path, f'{reg}_sd_delays'))
        plt.show()

        #print(f'channels with SD: {indices}')
        #print(f'mean: {np.nanmean(wave_delays[indices])} ms')
    return wave_delays, ch_with_sd
if __name__ == '__main__':
    plot_determine_onset_regions(path=os.getcwd())