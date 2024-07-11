import os 
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
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

            X_train, y_train,X_Test, y_test = (train_array[:,:-1],train_array[-1],test_array[:,:-1],test_array[-1])

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
                "CatBoost Regressor":CatBoostRegressor(),
                "Gradient Boost Regressor":GradientBoostingRegressor(),
            }

            
        except Exception as e:
            logging.error(f"Custom Exception ; {e}")
            raise CustomException(e,sys)


