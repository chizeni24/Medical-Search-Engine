
import pandas as pd

## read the csv file
def read_data(file_path, **kwargs):
    df = pd.read_csv(file_path  ,**kwargs)   #for preprocessing
    df1 = pd.read_csv(file_path  ,**kwargs)  #for returning results
    return df.iloc[:100,:]
