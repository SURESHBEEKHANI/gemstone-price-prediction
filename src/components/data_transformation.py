# Import necessary libraries

# sys - Used for handling system-specific functions, such as catching errors.
import sys

# dataclass - This helps in creating simple classes for holding data in a structured way.
from dataclasses import dataclass

# numpy (np) - A library for handling numerical data and performing mathematical operations.
import numpy as np 

# pandas (pd) - A powerful library for working with data, such as reading CSV files and manipulating tables.
import pandas as pd

# sklearn - A machine learning library that provides tools for data processing and model building.
from sklearn.compose import ColumnTransformer  # Helps apply different transformations to different columns
from sklearn.impute import SimpleImputer  # Used to fill missing values in the data
from sklearn.pipeline import Pipeline  # Allows us to chain several transformations together
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  # For encoding categorical data and scaling numerical data

# CustomException - A custom way to handle errors that may arise during data processing.
from ..exception import CustomException

# logging - A tool for tracking the program's activity, useful for debugging and understanding what's happening.
from src.logger import logging

# os - Helps in interacting with the operating system, like working with file paths.
import os

# save_object - A utility function for saving objects like the preprocessor configuration to disk.
from src.utils import save_object

# DataTransformationConfig is a simple class for storing the file path where the preprocessor will be saved.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# DataTransformation class is responsible for transforming the raw data into a format suitable for machine learning.
class DataTransformation:
    
    # The constructor initializes the configuration for data transformation
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # This function prepares the transformation process for both numerical and categorical columns
    def get_data_transformation_object(self):
        '''
        This function is responsible for defining how the data will be transformed (cleaned and processed).
        It sets up pipelines for numerical and categorical features, applying specific transformations.
        '''
        try:
            # Define the columns that are categorical (labels like 'cut', 'color', 'clarity')
            categorical_cols = ['cut', 'color', 'clarity']
            
            # Define the columns that are numerical (numbers like 'carat', 'depth', 'x', 'y', 'z')
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking (order) for categorical variables
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            # Numerical pipeline - handles missing values and scales data for better performance
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Replaces missing values with the median of the column
                    ('scaler', StandardScaler())  # Scales the numerical values to standardize the data
                ]
            )

            # Categorical pipeline - handles missing values, encodes categorical data, and scales it
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fills missing values with the most frequent value
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),  # Encodes categories into numbers
                    ('scaler', StandardScaler())  # Scales the encoded values
                ]
            )

            # Log the column names for debugging
            logging.info(f'Categorical Columns: {categorical_cols}')
            logging.info(f'Numerical Columns: {numerical_cols}')

            # Combine both pipelines (numerical and categorical) into one preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),  # Apply num_pipeline to numerical columns
                    ('cat_pipeline', cat_pipeline, categorical_cols)  # Apply cat_pipeline to categorical columns
                ]
            )

            # Return the preprocessor object, which is a combination of transformations
            return preprocessor

        except Exception as e:
            # Log any errors that occur during the transformation setup
            logging.info('Exception occurred in Data Transformation Phase')
            raise CustomException(e, sys)

    # This function performs the actual transformation of data, applying the preprocessing steps to both train and test data
    def initate_data_transformation(self, train_path, test_path):
        try:
            # Read the training and testing data from the provided file paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            # Get the preprocessor object that was defined earlier
            preprocessing_obj = self.get_data_transformation_object()

            # The target column is 'price' which we want to predict, so we separate it from the input features
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']  # Drop 'price' and 'id' as they are not features for prediction

            # Separate features (input data) and target (output data) for training
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features and target for testing
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Apply the preprocessor transformations to both training and testing features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target values to form the final training and testing data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object so it can be used later during predictions
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            # Return the transformed training and testing data along with the saved preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Log any errors during the data transformation process
            logging.info('Exception occurred in initiate_data_transformation function')
            raise CustomException(e, sys)
