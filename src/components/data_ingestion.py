
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import read_data, clean_data
# from src.utils import read_data,clean_data
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)


def inisiate_data_ingestion():
    """
    In this fun first we will read the data.
    Then we will apply basic preprocessing like handling missing valeues removing duplicates and handling datatypes
    We can save the data we will get in the raw folder
    """
    raw_data_path=os.path.join("Data/Raw","raw.csv")

    logging.info("Getting the data")
    df=read_data("https://raw.githubusercontent.com/JahidHasan010/All-Dataset/refs/heads/main/flight_price%20(1).csv")

    logging.info("Apply data cleaning")

    df=clean_data(df=df)

    logging.info("saving the raw data")
    df.to_csv(raw_data_path,index=False)

    return df

if __name__=="__main__":
    inisiate_data_ingestion()

# python src/components/data_ingestion.py
