import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# This class is responsible for handling the prediction pipeline
class PredictPipeline:
    def __init__(self):
        # Constructor, currently does nothing when an object is created
        pass

    # Method to make predictions using the trained model
    def predict(self, features):
        try:
            # Define paths to the saved preprocessor and model
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            
            # Load the preprocessor and model using the custom 'load_object' function
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            
            # Transform the input features using the preprocessor (e.g., scaling)
            data_scaled = preprocessor.transform(features)
            
            # Use the model to predict the outcome based on the transformed data
            pred = model.predict(data_scaled)
            
            # Return the predicted result
            return pred
        
        except Exception as e:
            # Log an error message if an exception occurs during the prediction process
            logging.info('Exception occurred in prediction pipeline')
            # Raise a custom exception with the error details
            raise CustomException(e, sys)

# This class represents the custom input data for prediction
class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        # Store the input features as attributes of the object
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    # Method to convert the input data into a pandas dataframe
    def get_data_as_dataframe(self):
        try:
            # Create a dictionary with the input data organized by feature
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            
            # Convert the dictionary into a pandas dataframe
            df = pd.DataFrame(custom_data_input_dict)
            
            # Log a message indicating that the dataframe was successfully created
            logging.info('Dataframe Gathered')
            
            # Return the dataframe
            return df
        
        except Exception as e:
            # Log an error message if an exception occurs during the dataframe creation
            logging.info('Exception Occurred in prediction pipeline')
            # Raise a custom exception with the error details
            raise CustomException(e, sys)
