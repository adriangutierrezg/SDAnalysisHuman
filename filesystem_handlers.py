#!/usr/bin/env python3
#adriangutierrezg 

"""
"""
import os
import sys
from szs_path import SZSPATH

sys.path.insert(0, SZSPATH)

from seizureProcessing.utils.get_directories import *

def adjustPathToLatestPreprocessingStep(path, start='DS_data', stop='reref'):
    '''adjusts path to the latest 
    stage of preprocessing
    
    Params:
    path, str or path object: path to session data
    stop, str: where in the path to stop looking
    
    Returns:
    path, os.path(): path to latest preprocessing stage
    '''
    
    a = start
    o = stop
    
    if path.endswith('/'):
        path = os.path.dirname(path) #fix path parsing issue
        
    #look for paths
    path_ = find_dir(path, a)
    
    if path_:
        path = path_
        del path_
    
    path_ = find_dir(path, o)
    
    if path_:
        path = path_
        del path_
    
    return path


if __name__ == '__main__':
    adjustPathToLatestPreprocessingStep()