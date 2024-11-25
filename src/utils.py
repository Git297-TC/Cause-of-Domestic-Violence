# Utility functions
#def helper():
    #print('Helper function')

# Import
import urllib
import urllib.request
import imblearn
import joblib
import matplotlib
import pandas
import pathlib
import seaborn
import sklearn
import time
import zipfile

# pandas setting
pandas.set_option('display.max_colwidth', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

def dataset_url_download(url, path):
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False
    
    if pathlib.Path(path).exists() == False:
        return False
    
    filename = path + 'rawDataset.zip'

    urllib.request.urlretrieve(url, filename)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(path)

    pathlib.Path.unlink(path + 'rawDataset.zip')

    return True

def dataset_csv_load2DataFrame(file_name, path, url):
    if pathlib.Path(path + file_name + '.csv').exists() == False:
        if dataset_url_download(url, path) == False:
            return
        
    dataframe = pandas.read_csv(path + file_name + '.csv')

    return dataframe

def dataframe_export_csv(dataframe, file_name, path):
    if pathlib.Path(path).exists() == False:
        return False
    
    dataframe.to_csv(path + file_name + '.csv', index=False)

    return True
