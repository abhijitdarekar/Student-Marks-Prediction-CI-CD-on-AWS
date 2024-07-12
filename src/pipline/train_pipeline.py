import os 
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluvate_model
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path :str = os.path.join("artifacts","model.pickle")

class ModelTrainer():
    def __init__(self) -> None:
        self.trained_model_file_path = ModelTrainerConfig()

    
    def initate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting Training and Training Data")

            X_train,y_train,X_Test, y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            logging.info(f"X Train Array {X_train.shape} y train Array{y_train.shape}")
            logging.info("Dictonary of Models")

            models = {
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "Linear Regression":LinearRegression(),
                "SVM Regression":SVR(),
                "KNN Regression":KNeighborsRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "XGBoost Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=0),
                "Gradient Boost Regressor":GradientBoostingRegressor(verbose=0),
            }

            model_report : dict = evaluvate_model( X_train, y_train,X_Test, y_test, models)
            logging.info(f"Model Report  {model_report}")
            # best_model_name = max(model_report)
            best_model_score = max(model_report.values())
            best_model_name = [key for key,v in model_report.items() if v ==best_model_score][0]
            best_Model = models[best_model_name]
            logging.info(f"Best Model {best_model_name}, has Score {best_model_score}")
            if best_model_score<0.6:
                raise CustomException("Model Performance : NO Best Model Found",sys)
            logging.info("MetricsBest Model Found on both train and test dataset.")

            save_object(
                file_path=self.trained_model_file_path.trained_model_file_path,
                object = best_Model
            )

            # Predicting 
            predicted = best_Model.predict(X_Test)
            r2_value = r2_score(y_test,predicted)
            return r2_value
            
        except Exception as e:
            logging.error(f"Custom Exception ; {e}")
            raise CustomException(e,sys)


