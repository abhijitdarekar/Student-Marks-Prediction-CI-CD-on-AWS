import os 
import sys
import numpy as np
import pandas as pd
import pickle


from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path, object):
    """
    Function to save the `object` in the `filepath`.
    """
    try:
        dir_path = os.path.dirname(file_path,)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(object,file)
    
    except Exception as e:
        logging.error(f"Custom Exception ; {e}")
        raise CustomException(e,sys)
    

def evaluvate_model( X_train, y_train,X_Test, y_test, models,params):
    """
    Function to Train the model and return the report.
    """

    try:

        report = {}
             
        for model in models.items():
            
            logging.info(f"MODEL TRAINING | Training Model {model[0]}")
            param = params[model[0]]
            grid_search_cv = GridSearchCV(model[1],param_grid=param,verbose=1,cv=3,)
            # modell = model[1]
            
            grid_search_cv.fit(X_train, y_train)
            modell = model[1]
            logging.info(f"Grid Search CV Model {model[0]}, Best Params {grid_search_cv.best_params_}")
            modell.set_params(**grid_search_cv.best_params_)
            modell.fit(X_train,y_train)

            y_pred_train = modell.predict(X_train)
            y_pred_test = modell.predict(X_Test)
            train_model_score = r2_score(y_train,y_pred_train)

            test_model_score = r2_score(y_test, y_pred_test)
            logging.info(f"{model[0]} | Training R2 : {train_model_score} | Testing R2 : {test_model_score}")
            report[model[0]]=test_model_score
            

        return report

    except Exception as e:
        logging.error(f"Custom Exception : {e}")
        raise CustomException(e,sys)
    

def load_object(object_path):
    try :
        with open(object_path,"rb") as file:
            obj = pickle.load(file)

            return obj
    except Exception as e:
        logging.error(f"Custom Exception : {e}")
        raise CustomException(e,sys)