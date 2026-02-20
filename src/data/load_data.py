import pandas as pd 
from src.config import RAW_DATA_PATH
def load_raw_data():
    """"
    load raw data set from config path

    return:
          pd.dataframe: raw dataset
    """
    try:
        df= pd.read_csv(RAW_DATA_PATH)
        return df
    except FileNotFoundError:
        raise Exception(f"File not found at path: {RAW_DATA_PATH}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data :{str(e)}")