import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
def save_object(file_path, obj):
    """
    Saves an object to a file using pickle.
    
    Parameters:
    - file_path: str, path where the object will be saved.
    - obj: object, the object to be saved.
    
    Raises:
    - CustomException: if there is an error during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info("Preprocessor object created successfully.")
        with open (file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info(f"Preprocessor object saved at {file_path}.")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)