# ======================================================
# utils.py
# Utility helper functions used across project
# ======================================================

import os
import sys
import dill                    # used to serialize python objects

from src.exception import CustomException
from src.logger import logging



# ======================================================
# Save object as pickle file
# Used for:
#   - saving model.pkl
#   - saving preprocessor.pkl
#   - saving any trained object
# ======================================================
def save_object(file_path, obj):

    try:
        # ----------------------------------------------
        # Get directory path from file path
        # Example: artifacts/model.pkl → artifacts/
        # ----------------------------------------------
        dir_path = os.path.dirname(file_path)

        # Create folder if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        logging.info(f"Saving object to {file_path}")

        # ----------------------------------------------
        # Save object using dill
        # dill can serialize almost anything (better than pickle)
        # ----------------------------------------------
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Object saved successfully")


    except Exception as e:
        raise CustomException(e, sys)



# ======================================================
# Load object from pickle file
# Used for:
#   - loading model for prediction
#   - loading preprocessor
# ======================================================
def load_object(file_path):

    try:
        logging.info(f"Loading object from {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Object loaded successfully")

        return obj


    except Exception as e:
        raise CustomException(e, sys)
