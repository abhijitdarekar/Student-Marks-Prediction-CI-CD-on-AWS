import os 
import sys

from src.logger import logging
from src.exception import CustomException

from pydantic import BaseModel,Field
import pandas as pd
from src.utils import load_object

import json

class StudentModelInput(BaseModel):
    gender : str = Field(description="The price must be greater than zero")
    race_ethnicity : str =Field(description="The price must be greater than zero")
    parental_level_of_education :str= Field(description="The price must be greater than zero")
    lunch : str= Field(description="The price must be greater than zero")
    test_preparation_course : str = Field(description="The price must be greater than zero")
    writing_score : int
    reading_score : int

class PredictPipeline:
    def __init__(self):
        self.model_object_path = "artifacts/model.pickle"
        self.preprocessor_object_path = "artifacts/Preprocessor.pickle"
        self.model = None
        self.preprocess = None

    def load_pickle_files(self,):
        logging.info("Prediction pipeline | Loading Model Object and PreProcessor Object file")
        try:
            self.model = load_object(self.model_object_path)
            self.preprocessor = load_object(self.preprocessor_object_path)
            logging.info("Prediction pipeline | Pre-Processor , Model loaded Sucessfully.")
        except Exception as e:
            logging.error(f"Exception : {e}")
            raise CustomException(e,sys)

    def get_data_as_Dataframe(self,json_data):
        df = pd.json_normalize(json.loads(json_data))
        logging.info("Prediction pipeline | Convered json to Dataframe")
        return df

    def predict(self, json_data):
        try:           
            df = self.get_data_as_Dataframe(json_data)
            data_scaled = self.preprocessor.transform(df)
            
            prediction = self.model.predict(data_scaled)
            return prediction
        except Exception as e:
            logging.error(f"Exception : {e}")
            raise CustomException(e,sys)