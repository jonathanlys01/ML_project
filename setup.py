"""
This file will format the data and create two csv files in the DATA_FOLDER.

This must only be run once, at the very beginning of the project.
"""
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

data_folder = os.getenv("DATA_FOLDER")
if data_folder == "":
    data_folder = os.path.join(os.getcwd(), "data")
    

print("Saving data to: " + data_folder)

while True:
    print("Continue? (y/n)")
    answer = input()
    if answer == "y":
        break
    elif answer == "n":
        exit()
        
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    
if os.environ.get("BANK_DATA_PATH") == "" or  os.environ.get("KIDNEY_DATA_PATH") == "":
    print("Please set the environment variables BANK_DATA_PATH and KIDNEY_DATA_PATH")
    exit()
    
print("Formatting data...")

bank_ds = pd.read_csv(os.environ.get("BANK_DATA_PATH"), sep=",", header=None)
bank_ds.columns = ["variance", "skewness", "curtosis", "entropy", "class"] # missing column names

kidney_ds = pd.read_csv(os.environ.get("KIDNEY_DATA_PATH"), index_col=0)
kidney_ds.rename(columns={"classification": "class"}, inplace=True)
# consistent target names

bank_ds.to_csv(os.path.join(data_folder, "bank.csv"), index=False)
kidney_ds.to_csv(os.path.join(data_folder, "kidney.csv"), index=False)

print("Setup complete.")
