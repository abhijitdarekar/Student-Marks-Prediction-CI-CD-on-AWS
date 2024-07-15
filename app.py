from src.pipline.predict_pipeline import StudentModelInput ,PredictPipeline

from fastapi import FastAPI
import os 
import sys

import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

logging.info("API SERVER | Started API Server")
predictionPipeline = PredictPipeline()
predictionPipeline.load_pickle_files()

app = FastAPI(title="Student Mark Prediction")

@app.get("/")
def root():
    logging.info("API SERVER | Root Page")
    return {"Status":"OK"}


@app.post("/predict",)
def predict(student_data:StudentModelInput):
    logging.info(f"API SERVER | Incoming Student Data : {student_data}")
    prediction = predictionPipeline.predict(student_data.model_dump_json())
    logging.info(f"API Server | Prediction output : {prediction}")
    return {'math_score':prediction[0]}