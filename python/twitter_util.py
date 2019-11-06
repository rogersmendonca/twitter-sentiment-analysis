from datetime import datetime,timedelta,date
import os
import sys

import langdetect as ld
import pandas as pd

def adjust_path_sep(path):
    return path.replace(os.path.altsep, os.path.sep)

def date_to_str(d):
    return d.strftime('%Y%m%d')

def now():
    return datetime.now().strftime('%d/%m/%Y %H:%M:%S')

def yyyymmdd_to_date(s):
    return date(int(s[0:4]),int(s[4:6]),int(s[6:8]))

def daterange(start_date,end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(days=n)
        
def detect_langs(string,file=sys.stdout):
    try:
        langs = ld.detect_langs(string.lower())
        return langs
    except:
        return None

def detect(string,file=sys.stdout):
    try:
        return ld.detect(string.lower())
    except:
        return None
    
def is_english(dict_langs):
    if (dict_langs is not None):
        for item in dict_langs:
            if item.lang == 'en':
                return True
    return False

def drop_duplicates_reindex(df):
    return df.drop_duplicates().reset_index(drop=True)

def strip_val(val):
    return str(val).strip() if (pd.notnull(val) and pd.notna(val)) else ''