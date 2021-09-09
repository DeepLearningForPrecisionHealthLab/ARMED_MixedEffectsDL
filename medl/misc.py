import datetime

def get_datestamp(with_time=False):
    now = datetime.datetime.now()
    strDate = now.strftime('%Y-%m-%d')
    
    if with_time:
        strDate += '_' + now.strftime('%H-%M-%S')
        
    return strDate