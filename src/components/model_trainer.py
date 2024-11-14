# Basic Import
import numpy as np  # This is a library that helps in doing mathematical operations and handling data.
import pandas as pd  # This is a library used to organize and analyze data in tables (like Excel).

# Modelling: These are different types of models (tools) we use to make predictions based on the data.
from sklearn.neighbors import KNeighborsRegressor  # A model that makes predictions based on similar past data.
from sklearn.tree import DecisionTreeRegressor  # A model that uses a tree-like structure to make predictions.
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor  # More advanced models that combine many simple models for better accuracy.
from sklearn.svm import SVR  # A model that uses support vector machines for predictions.
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # These are simpler models that predict based on linear relationships.
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV  # These help us fine-tune the models to make them work better.
from catboost import CatBoostRegressor  # A powerful machine learning tool designed to give good results quickly.
from xgboost import XGBRegressor  # Another advanced model similar to CatBoost that is popular for predictions.
from sklearn.ensemble import VotingRegressor  # A model that combines the results of multiple models for better accuracy.
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # These help us measure how good our predictions are.

# Custom tools we have created to handle errors, save models, and print results.
from src.exception import CustomException  # This handles any errors that occur during the process.
from src.logger import logging  # This records everything happening during the process.
from src.utils import save_object  # This saves the trained model so we can use it later.
from src.utils import evaluate_models  # This checks how well each model is working.
from src.utils import print_evaluated_results  # This prints the evaluation results of the model.
from src.utils import model_metrics  # This calculates how good the model's predictions are.

# Extra tools for handling data and file paths.
from dataclasses import dataclass  # This is used to organize information in a neat way.
import sys  # This is used to handle system-level functions.
import os  # This is used to handle file and folder paths on your computer.

# Configuration: This defines where we will save our trained model (after the model is ready).
@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')  # Location to save the model file.

# Main class that will train the model and make predictions.
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Setting up where to save the trained model.

    # Function to train and evaluate different models based on input data.
    def initate_model_training(self, train_array, test_array):
        try:
            # Splitting the data into features (inputs) and target (what we want to predict)
            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:, :-1],  # Input features (all columns except last one) for training data
                train_array[:, -1],   # The target value (last column) for training data
                test_array[:, :-1],   # Input features for testing data
                test_array[:, -1]     # The target value for testing data
            )
            
            # List of different models (tools) we will use to make predictions
            models = {
                "Linear Regression": LinearRegression(),  # Simple model
                "Lasso": Lasso(),  # Another simple model
                "Ridge": Ridge(),  # Another linear model
                "K-Neighbors Regressor": KNeighborsRegressor(),  # Model that looks for similar past examples
                "Decision Tree": DecisionTreeRegressor(),  # A tree-like model for predictions
                "Random Forest Regressor": RandomForestRegressor(),  # A model that uses many decision trees
                "XGBRegressor": XGBRegressor(),  # A powerful model used for large data
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),  # Fast and efficient model
                "GradientBoosting Regressor": GradientBoostingRegressor(),  # Another advanced model
                "AdaBoost Regressor": AdaBoostRegressor()  # A model that focuses on improving weak models
            }

            # Evaluating how well each model works on the given data
            model_report = evaluate_models(xtrain, ytrain, xtest, ytest, models)
            print(model_report)
            logging.info(f'Model Report : {model_report}')

            # Finding the model that performed the best based on its score (how accurate it is)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # If no model gives good enough results, we stop and raise an error
            if best_model_score < 0.6:
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            
            print(f'Best Model Found: {best_model_name} with R2 Score: {best_model_score}')
            logging.info(f'Best Model Found: {best_model_name} with R2 Score: {best_model_score}')

            # Tuning (improving) the parameters for the best model (CatBoost)
            logging.info('Hyperparameter tuning started for Catboost')
            cbr = CatBoostRegressor(verbose=False)

            # Defining different settings to improve the CatBoost model
            param_dist = {
                'depth': [4,5,6,7,8,9,10],
                'learning_rate': [0.01, 0.02, 0.03, 0.04],
                'iterations': [300, 400, 500, 600]
            }

            # Fine-tuning the CatBoost model using randomized search
            rscv = RandomizedSearchCV(cbr, param_dist, scoring='r2', cv=5, n_jobs=-1)
            rscv.fit(xtrain, ytrain)

            print(f'Best CatBoost Parameters: {rscv.best_params_}')
            print(f'Best CatBoost Score: {rscv.best_score_}')
            logging.info('Hyperparameter tuning complete for Catboost')

            # Now, tuning the KNN model (another prediction model)
            logging.info('Hyperparameter tuning started for KNN')
            knn = KNeighborsRegressor()
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)

            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid.fit(xtrain, ytrain)

            print(f'Best KNN Parameters: {grid.best_params_}')
            print(f'Best KNN Score: {grid.best_score_}')
            logging.info('Hyperparameter tuning complete for KNN')

            # Combining the best models into one final model (Voting Regressor)
            logging.info('Voting Regressor model training started')
            er = VotingRegressor([('cbr', rscv.best_estimator_), ('xgb', XGBRegressor()), ('knn', grid.best_estimator_)], weights=[3, 2, 1])
            er.fit(xtrain, ytrain)

            print('Final Model Evaluation:')
            print_evaluated_results(xtrain, ytrain, xtest, ytest, er)
            logging.info('Voting Regressor Training Completed')

            # Saving the trained model to a file so we can use it later
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=er)
            logging.info('Model pickle file saved')

            # Evaluating the final model's performance on the test data
            ytest_pred = er.predict(xtest)
            mae, rmse, r2 = model_metrics(ytest, ytest_pred)

            logging.info(f'Test MAE: {mae}')
            logging.info(f'Test RMSE: {rmse}')
            logging.info(f'Test R2 Score: {r2}')
            logging.info('Final Model Training Completed')
            
            return mae, rmse, r2  # Returning the evaluation results

        except Exception as e:
            # If there is an error at any point, log it and stop
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
