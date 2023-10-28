import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def fill_na(df,columns,strategy="most"):
    temp = df.copy()
    for column in columns:
        if strategy=="most":
            most = temp[column].value_counts().index[0]
            temp[column] = temp[column].fillna(most)
        else:
            raise NotImplementedError(f"{strategy} not a valid strategy)")
    return temp

def train_test_split(df, train_split=0.7):
    assert 0<=train_split<=1

    indices = np.array(df.index)

    np.random.shuffle(indices)

    train_indices = indices[:int(train_split*len(indices))]
    test_indices = indices[int(train_split*len(indices)):]

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    return train_df, test_df

def normalize(df, scaler):
    temp = df.copy()
    temp = temp.drop(columns=["class"])
    
    col = temp.columns

    temp = pd.DataFrame(scaler.fit_transform(temp))

    temp.columns = col
    temp["class"] = df["class"]

    return temp, scaler

def main():
    data_path = os.environ.get("DATA_FOLDER") if os.environ.get("DATA_FOLDER") != "" else os.path.join(os.getcwd(), "data") 
    ds_list = os.environ.get("DS").split(",")
    ds_list = [os.path.join(data_path, ds) for ds in ds_list]
    dfs = {}
    for path in ds_list:
        df = pd.read_csv(path)
        dfs[path] = df
        
    for path in dfs:

        scaler = StandardScaler()
        df = dfs[path]

        print(path)
        
        to_encode = [column for column in df.columns if df[column].dtype == "object"]
        
        print("Columns to encode: " + str(to_encode))
        encoded = int_encoding(df, to_encode)
        to_fill = [column for column in encoded.columns if np.sum(encoded.isna()[column])]
        print("Columns to fill: " + str(to_fill))
        filled = fill_na(encoded, to_fill)
        normalized, scaler = normalize(filled, scaler=scaler)

        print("encoded")
        print(encoded.head())

        print("filled")
        print(filled.head())

        print("normalized")
        print(normalized)

        print(np.mean(normalized, axis=0), np.std(normalized, axis =0))

        print("about the scaler")
        print("mean" ,scaler.mean_)
        print("scale", scaler.scale_)





        
if __name__ == "__main__":
    main()