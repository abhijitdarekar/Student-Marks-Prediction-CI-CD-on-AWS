import os 
import sys
import numpy as np
import pandas as pd
import dill


from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score



def save_object(file_path, object):
    """
    Function to save the `object` in the `filepath`.
    """
    try:
        dir_path = os.path.dirname(file_path,)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file:
            dill.dump(object,file)
    
    except Exception as e:
        logging.error(f"Custom Exception ; {e}")
        raise CustomException(e,sys)
    

def evaluvate_model( X_train, y_train,X_Test, y_test, models):
    """
    Function to Train the model and return the report.
    """

    try:

        report = {}

        for model in models.items():
            
            logging.info(f"MODEL TRAINING | Training Model {model[0]}")
            modell = model[1]
            modell.fit(X_train, y_train)
            y_pred_train = modell.predict(X_train)
            y_pred_test = modell.predict(X_Test)
            train_model_score = r2_score(y_train,y_pred_train)

            test_model_score = r2_score(y_test, y_pred_test)
            logging.info(f"{model[0]} | Training R2 : {train_model_score} | Testing R2 : {test_model_score}")
            report[model[0]]=test_model_score


        return report

    except Exception as e:
        logging.error(f"Custom Exception ; {e}")
        raise CustomException(e,sys)