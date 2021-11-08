import os
import datetime
from .settings import RESULTSDIR, DATADIR

def get_datestamp(with_time=False):
    now = datetime.datetime.now()
    strDate = now.strftime('%Y-%m-%d')
    
    if with_time:
        strDate += '_' + now.strftime('%H-%M-%S')
        
    return strDate

def expand_results_path(path, make=False):
    if path.startswith(os.path.sep):
        newpath = path
    else:
        newpath = os.path.join(RESULTSDIR, path)
    if os.path.exists(newpath):
        print('Warning: output path already exists')
    else:
        if make:
            os.makedirs(newpath)
    return newpath

def expand_data_path(path, make=False):
    if path.startswith(os.path.sep):
        newpath = path
    else:    
        newpath = os.path.join(DATADIR, path)       
    
    if not os.path.exists(newpath):
        if make:
            os.makedirs(newpath)
        else:
            print('Warning: data directory does not exist')
    
    return newpath