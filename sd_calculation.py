#!/usr/bin/env python3
#adriangutierrezg 

'''
'''

import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_wave_delays_per_wire(data_in_reg, onset_adjust_ms=0, stop_ms=2000, threshold_time_ms=1000, 
                            lpf_cutoff_fr_hz=3, filter_order=2, sf=1000):
    '''
    '''
    #initiate output array
    wave_delays = np.empty(len(data_in_reg))
    wave_delays.fill(np.nan)
    indices = []
    #calculate min and max of bundle
    #bundle_mean = np.mean([d[onset_adjust_ms:stop_ms]]
    minmax_data = np.asarray([abs(d[onset_adjust_ms:stop_ms]-np.mean(d[onset_adjust_ms:stop_ms])) 
                             for d in data_in_reg]).flatten()
    threshold = (max(minmax_data)-min(minmax_data))*0.5
    
    #iterate by wire
    for ix, wire in enumerate(data_in_reg):
        curr_data = wire[onset_adjust_ms:stop_ms] #cut data of interest
        filt = butter_lowpass_filter(curr_data, lpf_cutoff_fr_hz, sf, 
                                     order=filter_order)#filter data
        
        
        
        diff = np.diff(filt, n=1) #get first derivative of filtered data
        max_diff = abs(diff).argmax() + onset_adjust_ms #get abs maximum difference value 
        
        #make decision to add value
        amp_data = max(curr_data - np.mean(curr_data))-min(curr_data - np.mean(curr_data))*0.5
        #print(amp_data, threshold)
        if max_diff <= int(stop_ms/2):
            
            if amp_data >= threshold and (max(abs(curr_data))>250):
                wave_delays[ix] = max_diff
                indices.append(ix)
    
    return wave_delays, indices

if __name__ == '__main__':
    get_wave_delays_per_wire(data_in_reg)