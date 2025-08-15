import os
import sys
import numpy as np
import dill
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves an object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}.")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple regression models and returns their performance metrics.
    """
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            report[model_name] = {'rmse': rmse, 'r2_score': r2}
            logging.info(f"{model_name} - RMSE: {rmse}, R2 Score: {r2}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads an object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}.")
        return obj
    except Exception as e:
        raise CustomException(e, sys)