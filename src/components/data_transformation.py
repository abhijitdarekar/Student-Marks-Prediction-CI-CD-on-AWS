import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Saving the Preprocessing pipeline in file.
    """
    preprocessor_obj_file_path : str = os.path.join("artifacts","Preprocessor.pickle")


class DataTransformation:
    def __init__(self,):
        self.data_transformation_config = DataTransformationConfig()

    def _get_data_transformer_object(self,):
        """
        Function to Transfrom the data.
        """
        try:
            numerical_features = ['writing_score','reading_score']
            catogerical_features  = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),#,with_mean=False)),
                    ("StandardScalar",StandardScaler())
                ]
            )
            logging.info("Standard Scaling of Numerical Completed.")

            catogerical_pipeline = Pipeline(
                steps=(
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("One Hot Encoder",OneHotEncoder()),
                    ("StandardScaling",StandardScaler(with_mean=False))
                )
            )
            logging.info("Catogerical Columns Encoding Completed")

            pre_processor = ColumnTransformer(
                [
                    ("Numerical Column",num_pipeline,numerical_features),
                    ("Catogerical Columns",catogerical_pipeline,catogerical_features)
                ]
            )
            return pre_processor

        except Exception as e:
            logging.error(f"Data Transfromation Object Exception ; {e}")
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)
            logging.info("Completed | Importing Train and Test Data.")

            logging.info("Initilized | Obtaining pre-processor config")

            preprocessing_object = self._get_data_transformer_object()

            target_column_name = "math_score"
            numerical_features = ['writing_score','reading_score']
            catogerical_features  = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Processing Object to Dataframes.")

            input_feature_train_array = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_object.fit_transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            logging.info("Saving Processed Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object = preprocessing_object
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            logging.error(f"Initate Data Transformation Exception ; {e}")
            raise CustomException(e,sys)
        
