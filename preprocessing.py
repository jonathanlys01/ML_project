import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler

load_dotenv()

def int_encoding(df, columns):
    """
    Encode categorical variables as integers

    Parameters

    df: pandas.DataFrame
    columns: list (or str for one column) of column names to be encoded

    Returns

    df: pandas.DataFrame with encoded columns (original columns are not modified, a copy is returned)
    """
    temp = df.copy()
    if isinstance(columns, str):    
        columns = [columns]
    for column in columns:
        length = len(temp[column].value_counts())
        temp[column] = temp[column].map(dict(zip(temp[column].unique(), range(length))))
    return temp

def fill_na(df : pd.DataFrame,columns,strategy="median"):
    """
    Fill missing values in columns with a given strategy

    Parameters

    df: pandas.DataFrame
    columns: list (or str for one column) of column names to be encoded
    strategy: str, one of "most", "mean", "median", "zero" : strategy to fill missing values
    """
    temp = df.copy()
    for column in columns:
        sample = temp[column].dropna()
        if strategy=="most":
            most = sample.value_counts().index[0]
            temp[column].replace(np.nan, most, inplace=True)
        elif strategy=="mean": # warning : not recommended for one hot encoded categorical variables
            mean = np.mean(sample)
            temp[column].replace(np.nan, mean, inplace=True)
        elif strategy=="median": 
            median = np.median(sample)
            temp[column].replace(np.nan, median, inplace=True)
        elif strategy=="zero":
            temp[column].replace(np.nan, 0, inplace=True)
        else:
            raise NotImplementedError(f"{strategy} not a valid strategy)")
    return temp

def train_test_split(df:pd.DataFrame, train_split=0.7):
    """
    Split a dataframe into train and test sets

    Parameters

    df: pandas.DataFrame
    train_split: float, proportion of the dataset to be used for training

    Returns

    train_df: pandas.DataFrame, train set
    test_df: pandas.DataFrame, test set
    """
    assert 0<=train_split<=1

    indices = np.array(df.index)

    np.random.shuffle(indices)

    train_indices = indices[:int(train_split*len(indices))]
    test_indices = indices[int(train_split*len(indices)):]

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    return train_df, test_df

def normalize(df, scaler, fitted=False):
    """
    Normalize a dataframe using a given scaler

    Parameters

    df: pandas.DataFrame
    scaler: sklearn.preprocessing scaler
    fitted: bool, if True, fit the scaler on the dataframe

    Returns

    df: pandas.DataFrame, normalized dataframe
    scaler: sklearn.preprocessing scaler, fitted scaler
    for example, a StandardScaler will have fitted mean and standard deviation (scaler.mean_, scaler.scale)
    """
    temp = df.copy()
    temp = temp.drop(columns=["class"])
    
    col = temp.columns

    if fitted:
        scaler.fit(temp)

    temp = pd.DataFrame(scaler.transform(temp))

    temp.columns = col
    temp.index = df.index
    temp["class"] = df["class"].copy()

    return temp, scaler

def get_X_y(df:pd.DataFrame):
    """
    Split a dataframe into features and target

    Parameters

    df: pandas.DataFrame

    Returns

    X: numpy.ndarray, features
    y: numpy.ndarray, target
    """
    X = df.drop(columns=["class"]).to_numpy()
    y = df["class"].to_numpy()
    return X, y

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