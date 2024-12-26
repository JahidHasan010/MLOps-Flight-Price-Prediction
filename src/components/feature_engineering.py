
"""
This script is responsible for feature engnering.
First we can get the data and do feature engnering
After Feature engnering we can save the full clean data in the process folder.
We will also split the data into train and test file and also save them into process folder.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
logging.basicConfig(level=logging.INFO)

class Feature_engnering_config:
    def __init__(self,df):
        self.df=df
        self.process_data_path=os.path.join('Data/Process',"clean.csv")
        self.train_data_path=os.path.join('Data/Process',"train.csv")
        self.test_data_path=os.path.join('Data/Process',"test.csv")

    # def initiate_feature_engineering(self):
    #     """
    #     1- First we will convert the duration to min.
    #     2- Second we will convert to total stop into interger format
    #     3- Third we will get the dep_hour and arrival hourr and similary dep_min and arrival_min
    #     4- We can also saperate the month,year and day from journey date.
    #     5- We can rename the business_arlines to normal airlines
    #     6- We will remove unecessary columns and get the necessary columns
    #     7- There is airline that contain 4 stop they are outlier we can remove this
    #     """

    #     # Covert the Duration in minutes
    #     logging.info("Convert the total duration into minutes")
    #     self.df["Duration_In_Min"]=self.convert_to_min('Duration')

    #     # Covert the total stop into int format
    #     logging.info("Covert the total stops into integer format")
    #     self.df['Total_Stops']=self.df['Total_Stops'].str.extract(r"(\d+)").fillna(0).astype(int)
        
    #     # Get the the deperture hour and depertue min
    #     logging.info("Getting dep hour and min")
    #     self.df['Dep_hour']=pd.to_datetime(self.df['Dep_Time']).dt.hour
    #     self.df['Dep_min']=pd.to_datetime(self.df['Dep_Time']).dt.minute

    #     # Get the the arrival hour and arrival min
    #     logging.info("Getting arrival hour and min")
    #     self.df['Arrival_hour']=pd.to_datetime(self.df['Arrival_Time']).dt.hour
    #     self.df['Arrival_min']=pd.to_datetime(self.df['Arrival_Time']).dt.minute

    #     # Saperate the month year and day
    #     logging.info('saperate year month and day from journey col')
    #     self.df["Journey_year"]=self.df['Date_of_Journey'].dt.year
    #     self.df['Journey_month']=self.df['Date_of_Journey'].dt.month_name()
    #     self.df['Journey_day']=self.df['Date_of_Journey'].dt.day_name()

    #     # Rename the bussiness airlines to normal airlines
    #     logging.info("Renaming the bussiess class to normal class airlines")
    #     self.df['Airline']=np.where(
    #             self.df['Airline']=='Vistara Premium economy','Vistara',
    #             np.where(self.df['Airline']=='Jet Airways Business','Jet Airways',
    #                 self.df['Airline'])
    #         )

    #     # drop columns
    #     # Note we can also drop the journey year b/c t  contain only one single value.
    #     logging.info("Drop unecessary columns")
    #     col_to_drop=['Duration','Date_of_Journey','Arrival_Time',
    #                 'Dep_Time','Route','Additional_Info','Journey_year']
    #     self.df=self.drop_col(col_list=col_to_drop)

    #     # Removeing 4 stop value
    #     logging.info("Removing the row whcih contain 4 stops")
    #     self.df=self.df[~(self.df['Total_Stops']==4)]

    #     # Save the clean data in the process folder
    #     self.df.to_csv(self.process_data_path,index=False)
    #     logging.info(f"Clean data should be save in this location {self.process_data_path}")
        
    #     logging.info("Splititng the data")
    #     train_data,test_data=self.split_data()
    #     # Save the train and test data
    #     logging.info("Saving the train data")
    #     train_data.to_csv(self.train_data_path,index=False)
        
    #     logging.info("Saving the test data")
    #     test_data.to_csv(self.test_data_path,index=False)
        
    #     return [
    #         self.train_data_path,
    #         self.test_data_path
    #     ]

    # # new add

    # def initiate_feature_engineering(self):
    #     """
    #     1. Convert the duration to minutes.
    #     2. Convert total stops into integer format.
    #     3. Extract departure and arrival hours and minutes.
    #     4. Extract year, month, and day from journey date.
    #     5. Rename business airlines to standard airline names.
    #     6. Remove unnecessary columns.
    #     7. Remove rows with outliers (e.g., 4 stops).
    #     """
    #     # Convert the duration into minutes
    #     logging.info("Convert the total duration into minutes")
    #     self.df["Duration_In_Min"] = self.convert_to_min('Duration')

    #     # Convert the total stops into integer format
    #     logging.info("Convert the total stops into integer format")
    #     self.df['Total_Stops'] = self.df['Total_Stops'].str.extract(r"(\d+)").fillna(0).astype(int)

    #     # Extract departure hour and minute
    #     logging.info("Extracting departure hour and minute")
    #     self.df['Dep_Time'] = pd.to_datetime(self.df['Dep_Time'], format='%H:%M')
    #     self.df['Dep_hour'] = self.df['Dep_Time'].dt.hour
    #     self.df['Dep_min'] = self.df['Dep_Time'].dt.minute

    #     # Extract arrival hour and minute
    #     logging.info("Extracting arrival hour and minute")
    #     self.df['Arrival_Time'] = pd.to_datetime(self.df['Arrival_Time'], format='%H:%M')
    #     self.df['Arrival_hour'] = self.df['Arrival_Time'].dt.hour
    #     self.df['Arrival_min'] = self.df['Arrival_Time'].dt.minute

    #     # Extract year, month, and day from journey date
    #     logging.info("Extracting year, month, and day from journey date")
    #     self.df['Date_of_Journey'] = pd.to_datetime(self.df['Date_of_Journey'], format='%d/%m/%Y')
    #     self.df["Journey_year"] = self.df['Date_of_Journey'].dt.year
    #     self.df['Journey_month'] = self.df['Date_of_Journey'].dt.month
    #     self.df['Journey_day'] = self.df['Date_of_Journey'].dt.day

    #     # Rename business class airlines to standard class airlines
    #     logging.info("Renaming business class airlines")
    #     self.df['Airline'] = self.df['Airline'].replace({
    #         'Vistara Premium economy': 'Vistara',
    #         'Jet Airways Business': 'Jet Airways'
    #     })

    #     # Drop unnecessary columns
    #     logging.info("Dropping unnecessary columns")
    #     col_to_drop = ['Duration', 'Date_of_Journey', 'Arrival_Time', 'Dep_Time', 'Route', 'Additional_Info', 'Journey_year']
    #     self.df = self.df.drop(columns=col_to_drop)

    #     # Remove rows with outliers (e.g., 4 stops)
    #     logging.info("Removing rows with 4 stops")
    #     self.df = self.df[self.df['Total_Stops'] != 4]

    #     # Save the cleaned data
    #     logging.info(f"Saving cleaned data to {self.process_data_path}")
    #     self.df.to_csv(self.process_data_path, index=False)

    #     # Split the data
    #     logging.info("Splitting data into train and test sets")
    #     train_data, test_data = self.split_data()

    #     # Save train and test data
    #     logging.info(f"Saving train data to {self.train_data_path}")
    #     train_data.to_csv(self.train_data_path, index=False)

    #     logging.info(f"Saving test data to {self.test_data_path}")
    #     test_data.to_csv(self.test_data_path, index=False)

    #     return [self.train_data_path, self.test_data_path]

    def initiate_feature_engineering(self):
        """
        Perform feature engineering on the dataset.
        """
        try:
            # Convert the duration into minutes
            logging.info("Convert the total duration into minutes")
            self.df["Duration_In_Min"] = self.convert_to_min('Duration')

            # Convert the total stops into integer format
            logging.info("Convert the total stops into integer format")
            self.df['Total_Stops'] = self.df['Total_Stops'].str.extract(r"(\d+)").fillna(0).astype(int)

            # Extract departure hour and minute
            logging.info("Extracting departure hour and minute")
            self.df['Dep_Time'] = pd.to_datetime(self.df['Dep_Time'], errors='coerce', format='%H:%M')
            self.df['Dep_hour'] = self.df['Dep_Time'].dt.hour
            self.df['Dep_min'] = self.df['Dep_Time'].dt.minute

            # Extract arrival hour and minute
            logging.info("Extracting arrival hour and minute")
            # Coerce invalid formats to NaT and drop them
            self.df['Arrival_Time'] = pd.to_datetime(self.df['Arrival_Time'], errors='coerce', format='%H:%M')
            invalid_arrival_count = self.df['Arrival_Time'].isna().sum()
            logging.warning(f"Invalid Arrival_Time entries: {invalid_arrival_count}")
            self.df = self.df.dropna(subset=['Arrival_Time'])
            
            self.df['Arrival_hour'] = self.df['Arrival_Time'].dt.hour
            self.df['Arrival_min'] = self.df['Arrival_Time'].dt.minute

            # Extract year, month, and day from journey date
            logging.info("Extracting year, month, and day from journey date")
            self.df['Date_of_Journey'] = pd.to_datetime(self.df['Date_of_Journey'], errors='coerce', format='%d/%m/%Y')
            self.df['Journey_year'] = self.df['Date_of_Journey'].dt.year
            self.df['Journey_month'] = self.df['Date_of_Journey'].dt.month
            self.df['Journey_day'] = self.df['Date_of_Journey'].dt.day

            # Rename business class airlines to standard class airlines
            logging.info("Renaming business class airlines")
            self.df['Airline'] = self.df['Airline'].replace({
                'Vistara Premium economy': 'Vistara',
                'Jet Airways Business': 'Jet Airways'
            })

            # Drop unnecessary columns
            logging.info("Dropping unnecessary columns")
            col_to_drop = ['Duration', 'Date_of_Journey', 'Arrival_Time', 'Dep_Time', 'Route', 'Additional_Info', 'Journey_year']
            self.df = self.df.drop(columns=col_to_drop)

            # Remove rows with outliers (e.g., 4 stops)
            logging.info("Removing rows with 4 stops")
            self.df = self.df[self.df['Total_Stops'] != 4]

            # Save the cleaned data
            logging.info(f"Saving cleaned data to {self.process_data_path}")
            self.df.to_csv(self.process_data_path, index=False)

            # Split the data
            logging.info("Splitting data into train and test sets")
            train_data, test_data = self.split_data()

            # Save train and test data
            logging.info(f"Saving train data to {self.train_data_path}")
            train_data.to_csv(self.train_data_path, index=False)

            logging.info(f"Saving test data to {self.test_data_path}")
            test_data.to_csv(self.test_data_path, index=False)

            return [self.train_data_path, self.test_data_path]
        except Exception as e:
            logging.error(f"Error during feature engineering: {e}")
            raise e

    def drop_col(self,col_list):
        """
        This fun is responsible for drop the unecessary columns
        """
        self.df=self.df.drop(columns=col_list)
        return self.df


    def convert_to_min(self,col):
        """
        This fun can take the col and saperate the hour and min and then convert the hour to min.
        """
        try:
            hour=self.df[col].str.replace("h","").str.replace("m","").str.split().str.get(0).astype('int')
            minute=self.df[col].str.replace("h","").str.replace("m","").str.split().str.get(1).astype("float").fillna(0)
            
            return (hour * 60) + minute
        except Exception as e:
            return e
    
    def split_data(self):
        """
        This fun is responsible to split the data into train and test set
        """
        train_data,test_data=train_test_split(self.df,test_size=0.2,random_state=43)

        return train_data,test_data
    

if __name__ == "__main__":
    # Example usage
    # Load data (replace with actual path or DataFrame)
    df = pd.read_csv("Data/Raw/raw.csv")  # Replace with your data path
    fe = Feature_engnering_config(df=df)
    train_data_path, test_data_path = fe.initiate_feature_engineering()

# python src/components/feature_engineering.py
