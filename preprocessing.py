import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def int_encoding(df, columns):
    # columns: list (or str for one column) of column names to be encoded
    temp = df.copy()
    if isinstance(columns, str):    
        columns = [columns]
    for column in columns:
        length = len(temp[column].value_counts())
        temp[column] = temp[column].map(dict(zip(temp[column].unique(), range(length))))
    return temp

def main():
    data_path = os.environ.get("DATA_FOLDER") if os.environ.get("DATA_FOLDER") != "" else os.path.join(os.getcwd(), "data") 
    ds_list = os.environ.get("DS").split(",")
    ds_list = [os.path.join(data_path, ds) for ds in ds_list]
    dfs = {}
    for path in ds_list:
        df = pd.read_csv(path)
        dfs[path] = df
        
    for path in dfs:
        df = dfs[path]
        
        to_encode = [column for column in df.columns if df[column].dtype == "object"]
        print(path)
        print("Columns to encode: " + str(to_encode))
        print(int_encoding(df, to_encode))
        
if __name__ == "__main__":
    main()