# Import necessary libraries and modules

# os - This library helps to interact with the operating system, like working with files and directories.
import os  

# sys - Used for system-specific parameters, like handling exceptions in the code.
import sys  

# CustomException - A special way to handle errors specifically for this project.
from ..exception import CustomException

# logging - This is used to keep track of the program’s activity, like debugging or recording information.
from src.logger import logging  

# pandas (pd) - A library for handling data, especially large datasets, like spreadsheets or CSV files.
import pandas as pd  

# train_test_split - A function that helps to split data into two parts: training and testing datasets.
from sklearn.model_selection import train_test_split  

# dataclass - A Python feature used to create simple classes that are just containers for data.
from dataclasses import dataclass  

# Import components for data transformation and model training
# These are classes and configurations for transforming data and training models.
from src.components.data_transformation import DataTransformation  
from src.components.data_transformation import DataTransformationConfig  
from src.components.model_trainer import ModelTrainerConfig  
from src.components.model_trainer import ModelTrainer  

# Initialize Data Ingestion Configuration
@dataclass
# DataIngestionConfig is a simple class that holds the file paths for storing data.
class DataIngestionConfig:
    # Defining paths where we will save the raw, training, and testing data.
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    # The __init__ method is automatically called when an object of the class is created.
    # It initializes the data ingestion configuration.
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Method that initiates the data ingestion process
    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')  # Log to record when the process starts
        try:
            # Reading a CSV file (containing gemstone data) and storing it as a pandas dataframe.
            df = pd.read_csv('notebook/data/gemstone.csv')
            logging.info('Dataset read as pandas Dataframe')  # Log success message

            # Creating necessary directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Saving the raw data to the path defined in the configuration
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info('Train Test Split Initiated')  # Log when the split process starts
            # Splitting the dataset into two parts: training data (80%) and testing data (20%)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the training and testing data into separate files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')  # Log when the process finishes

            # Return the file paths for the training and testing data
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # In case of an error, log the error and raise an exception to handle it.
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Main block to execute the data ingestion process
if __name__ == '__main__':
    # Create an instance (object) of the DataIngestion class
    obj = DataIngestion()
    # Calling the method to ingest data (read and split it into training and testing)
    train_data, test_data = obj.initate_data_ingestion()

    # Create an object for DataTransformation class and transform the data (like cleaning or changing format)
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_data, test_data)

    # Create an object for ModelTrainer class and start training the model using the transformed data
    modeltrainer = ModelTrainer()
    modeltrainer.initate_model_training(train_arr, test_arr)
