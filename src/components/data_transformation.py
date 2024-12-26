
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_file
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

"""
- In this file i can do all the feature transformation
- Feature transformation like impute values, encoding scaling etc.
"""

class data_transformation_config:
    def __init__(self,train_data_path,test_data_path):
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
        self.processor_path=os.path.join("data/transformation","processor.pkl")
    
    def inisiate_data_transformation(self):
        logging.info("Reading the train and test data")
        train_data=pd.read_csv(self.train_data_path)

        test_data=pd.read_csv(self.test_data_path)
        logging.info("Reading train and test data successfully")
        
        # saperate feature and labels
        logging.info("saperate the feature and labels")
        x_train=train_data.drop(columns=['Price'])
        y_train=train_data['Price'].values.reshape(-1,1)

        x_test=test_data.drop(columns=['Price'])
        y_test=test_data['Price'].values.reshape(-1,1)

        # saperate num and categorical columns
        num_col=x_train.select_dtypes("number").columns
        cat_col=x_train.select_dtypes("object").columns

        # call the build pipeline fun
        processor=self.build_pipeline(cat_col=cat_col,num_col=num_col)
        
        # apply the processor on train and test data
        logging.info("applying the transformation on train and test data")
        x_train_transform=processor.fit_transform(x_train)
        x_test_transform=processor.transform(x_test)
        logging.info("Transformation done")

        # concate the transfrom data with output columns
        logging.info("Concate the train and test array")
        train_array=np.c_[x_train_transform,y_train]
        test_array=np.c_[x_test_transform,y_test]
        logging.info("concatination done")
        
        # Ensure the output directories exist
        os.makedirs("Data/transformation", exist_ok=True)

        logging.info("save the processor")
        save_file(obj=processor,file_path=self.processor_path)
        
        # Ensure the output directories exist
        # os.makedirs("Transformation", exist_ok=True)

        #   Optionally save the arrays
        np.save("data/transformation/train_array.npy", train_array)
        np.save("data/transformation/test_array.npy", test_array)
        logging.info("Train and test arrays saved successfully")

        return [
            train_array,
            test_array
        ]


    def build_pipeline(self,num_col,cat_col):
        # Build numerical pipeline
        logging.info("Build numerical pipeline")
        num_pipe=Pipeline(steps=[
            ("impute",SimpleImputer(strategy='median')),
            ("scale",StandardScaler())
        ])

        # build a categorical pipeline
        logging.info("Build a categorical pipeline")
        cat_pipe=Pipeline(steps=[
            ("impute",SimpleImputer(strategy="most_frequent")),
            ("encode",OneHotEncoder(drop="first",sparse_output=False,
                                    handle_unknown='ignore'))
        ])

        # Build a columns transformer
        processor=ColumnTransformer(transformers=[
            ("Num_transform",num_pipe,num_col),
            ("Cat_transform",cat_pipe,cat_col)
        ])


        return processor
    

    

if __name__=="__main__":
    
    # train_data_path =  df = pd.read_csv("Data/Process/train.csv")
    # test_data_path =  df = pd.read_csv("Data/Process/test.csv")
    # Data transformation
    # dt=data_transformation_config(train_data_path=train_data_path,
    #                                 test_data_path=test_data_path)
    # train_array,test_array=dt.inisiate_data_transformation()
    obj=data_transformation_config(train_data_path="data/process/train.csv",test_data_path="data/process/test.csv")
    obj.inisiate_data_transformation()

# # python src/components/data_transformation.py


# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# import pandas as pd
# import numpy as np
# import logging
# import os
# import pickle

# logging.basicConfig(level=logging.INFO)

# # logging.basicConfig(
# #     format="%(asctime)s - %(levelname)s - %(message)s",
# #     level=logging.INFO,
# #     handlers=[logging.FileHandler("logs/data_transformation.log"), logging.StreamHandler()]
# # )

# class DataTransformationConfig:
#     def __init__(self, train_data_path, test_data_path):
#         """
#         Initialize configuration for data transformation.
#         Args:
#             train_data_path (str): Path to the train dataset.
#             test_data_path (str): Path to the test dataset.
#         """
#         self.train_data_path = train_data_path
#         self.test_data_path = test_data_path
#         self.processor_path = os.path.join("Models", "processor.pkl")

#     def initiate_data_transformation(self):
#         """
#         Execute the full data transformation pipeline.
#         Returns:
#             list: Transformed train and test arrays.
#         """
#         try:
#             logging.info("Reading the train and test data")
#             train_data = pd.read_csv(self.train_data_path)
#             test_data = pd.read_csv(self.test_data_path)
#             logging.info("Train and test data loaded successfully.")

#             # Separate features and target
#             logging.info("Separating features and target labels")
#             x_train = train_data.drop(columns=['Price'])
#             y_train = train_data['Price'].values.reshape(-1, 1)

#             x_test = test_data.drop(columns=['Price'])
#             y_test = test_data['Price'].values.reshape(-1, 1)

#             # Identify numerical and categorical columns
#             num_col = x_train.select_dtypes(include=["number"]).columns
#             cat_col = x_train.select_dtypes(include=["object"]).columns
#             logging.info(f"Identified numerical columns: {list(num_col)}")
#             logging.info(f"Identified categorical columns: {list(cat_col)}")

#             # Build and apply the processor
#             logging.info("Building the transformation pipeline")
#             processor = self.build_pipeline(num_col=num_col, cat_col=cat_col)
#             logging.info("Applying transformations on train and test data")
#             x_train_transformed = processor.fit_transform(x_train)
#             x_test_transformed = processor.transform(x_test)

#             # Concatenate transformed data with target
#             logging.info("Concatenating transformed features with target labels")
#             train_array = np.c_[x_train_transformed, y_train]
#             test_array = np.c_[x_test_transformed, y_test]

#             # Save the processor
#             logging.info("Saving the processor object")
#             self.save_file(obj=processor, file_path=self.processor_path)

#             # Optionally save the arrays
#             np.save("artifacts/train_array.npy", train_array)
#             np.save("artifacts/test_array.npy", test_array)
#             logging.info("Train and test arrays saved successfully")

#             return [train_array, test_array]
#         except Exception as e:
#             logging.error(f"Error during data transformation: {e}")
#             raise

#     def build_pipeline(self, num_col, cat_col):
#         """
#         Build the data transformation pipeline.
#         Args:
#             num_col (list): Numerical columns.
#             cat_col (list): Categorical columns.
#         Returns:
#             ColumnTransformer: A pipeline for transforming the dataset.
#         """
#         try:
#             # Numerical pipeline
#             logging.info("Building numerical transformation pipeline")
#             num_pipeline = Pipeline(steps=[
#                 ("imputer", SimpleImputer(strategy='median')),
#                 ("scaler", StandardScaler())
#             ])

#             # Categorical pipeline
#             logging.info("Building categorical transformation pipeline")
#             cat_pipeline = Pipeline(steps=[
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'))
#             ])

#             # Column transformer
#             processor = ColumnTransformer(transformers=[
#                 ("num_transform", num_pipeline, num_col),
#                 ("cat_transform", cat_pipeline, cat_col)
#             ])

#             return processor
#         except Exception as e:
#             logging.error(f"Error during pipeline creation: {e}")
#             raise

#     @staticmethod
#     def save_file(obj, file_path):
#         """
#         Save a Python object to a file using pickle.
#         Args:
#             obj: Python object to save.
#             file_path (str): File path for saving the object.
#         """
#         try:
#             os.makedirs(os.path.dirname(file_path), exist_ok=True)
#             with open(file_path, "wb") as file:
#                 pickle.dump(obj, file)
#             logging.info(f"Object saved successfully at {file_path}")
#         except Exception as e:
#             logging.error(f"Error while saving object: {e}")
#             raise


# if __name__ == "__main__":
#     try:
#         # Define paths for train and test datasets
#         train_data_path = "Data/Process/train.csv"
#         test_data_path = "Data/Process/test.csv"

#         # Ensure the output directories exist
#         os.makedirs("artifacts", exist_ok=True)
#         # os.makedirs("logs", exist_ok=True)
#         os.makedirs("Models", exist_ok=True)

#         # Initiate the data transformation process
#         logging.info("Starting data transformation")
#         data_transformer = DataTransformationConfig(
#             train_data_path=train_data_path,
#             test_data_path=test_data_path
#         )
#         train_array, test_array = data_transformer.initiate_data_transformation()
#         logging.info("Data transformation completed successfully")
#     except Exception as e:
#         logging.error(f"Error in main execution: {e}")
#         raise
