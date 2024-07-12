import os 
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig

from src.pipline.train_pipeline import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _initate_data_ingestion(self):
        """
        Function to Fetch data from database, or pandas dataframe.
        """
        logging.info("Entered DataIngestion Component")
        try:
            # Fetching Data from DataFrame
            df = pd.read_csv(os.path.join("notebooks","data","student.csv"))
            logging.info("Imported Datset Complete.")

            # Createing Artifacts training , testinng, raw data 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Saving Raw Data
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train-Test Split Initated.")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)

            logging.info(f"Train Set Shape{train_set.shape},Test set Shape {test_set.shape}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Caught Ingestion :{e}")
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj._initate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    logging.info(f"Data Ingestion File | Train Array Size {train_array.shape},Test Array Shape {test_array.shape}")
    model_Trainer = ModelTrainer()
    print("Best R2 Value is :",model_Trainer.initate_model_trainer(train_array,test_array,None))
