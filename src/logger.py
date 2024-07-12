import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y')}.log" #%H_%M_%S')}.log"
LOG_FOLDER = f"{datetime.now().strftime('%m_%d_%Y')}"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FOLDER)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logging.info("\n\nStart of Logging -----------------------------------------------\n\n")
