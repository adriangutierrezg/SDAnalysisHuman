#!/usr/bin/env python3
#adriangutierrezg 

"""
"""
import pandas as pd
import os
import sys 
from scipy.io import loadmat
from szs_path import SZSPATH

from filesystem_handlers import *
sys.path.insert(0, SZSPATH) #local imports
from seizureProcessing.utils.get_directories import *
from seizureProcessing.utils.get_chnames import *
from seizureProcessing.utils.get_patinfo_from_path import *


def make_basic_dataframe(path, sf=1000, so_sec=30*60, prep_stage='reref'):
    rerefpath = adjustPathToLatestPreprocessingStep(path, stop=prep_stage)
    fnames = match_cssfiles_to_chnames(path)
    pnr, sznr = get_patinfo_from_path(path)
    chnames = get_chnames(path, strip=True)
    #matfiles_data = [x.strip('.ncs')+'_reref_data.mat' for x in fnames]
    matfiles_times = [x.strip('.ncs')+'_reref_times.mat' for x in fnames]
    
    #calculate reported seizure onset timestamp
    
    #sf = 1000
    times = loadmat(os.path.join(rerefpath,matfiles_times[0]))
    if 'reref' in rerefpath[-7:]:
        tstamps = times['reref_times'][0]
    elif 'DS_data' in rerefpath[-7:]:
        tstamps = times['DS_times'][0]
    
    
    so_index = int(so_sec*sf)
    reported_so_ts = tstamps[int(so_sec*sf)]

    df = pd.DataFrame()
    
    #add all other important data
    for i, ch in enumerate(chnames):
        df['pnr'] = pnr
        df['sznr'] = sznr
        df['datapath'] = rerefpath
        df['chname'] = chnames
        df['region'] = [x[:-1] for x in chnames]
        df['hemisphere'] = [x[0] for x in chnames]
        df['sf'] = sf
        #df['SO_ts'] = reported_so_ts
        df['SO_sec'] = so_sec
        #df['SO_index'] = so_index
        df['fname'] = fnames
    return df

def modifyValueToDataframe(df, val, col, key=None, key_col=None):
    '''modifies value to pandas dataframe
    if key=None, it adds value to all entries, else 
    it does it only to filtered (through key) dataframe'''
    
    if not col in df.keys().tolist():
        df[col] = ''*len(df)
    
    if type(key)==list:
        for k in key:
            df.loc[df[key_col]==k, [col]] = val
    else:
        df[col] = val

if __name__ == '__main__':
    make_basic_dataframe(path=os.getcwd())