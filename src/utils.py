# ======================================================
# utils.py
# Utility helper functions used across the project
# ======================================================

import os
import sys
import dill                          # used to serialize Python objects
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# ======================================================
# Function: save_object
# Purpose:
#   Save any Python object (model, preprocessor, etc.)
#   into a pickle file using dill
# ======================================================
def save_object(file_path, obj):

    try:
        # ----------------------------------------------
        # Extract directory path from full file path
        # Example:
        #   artifacts/model.pkl → artifacts/
        # ----------------------------------------------
        dir_path = os.path.dirname(file_path)

        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        logging.info(f"Saving object to {file_path}")

        # ----------------------------------------------
        # Serialize and save object using dill
        # dill is more flexible than pickle
        # ----------------------------------------------
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise CustomException(e, sys)



# ======================================================
# Function: load_object
# Purpose:
#   Load a previously saved object
#   (model, preprocessor, etc.)
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



# ======================================================
# Function: evaluate_models
# Purpose:
#   Train multiple models and compare performance
#
# Input:
#   - X_train, y_train
#   - X_test, y_test
#   - models (dictionary of model_name: model_object)
#
# Output:
#   - Dictionary of model_name : test_r2_score
# ======================================================
def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    try:
        report = {}

        for model_name, model in models.items():

            logging.info(f"Running GridSearch for {model_name}")

            param_grid = params.get(model_name, {})

            # If no hyperparameters provided
            if not param_grid:
                model.fit(X_train, y_train)
                best_model = model

            else:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=1
                )

                gs.fit(X_train, y_train)

                # Best already refit
                best_model = gs.best_estimator_

                logging.info(
                    f"{model_name} -> Best Params: {gs.best_params_}"
                )

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(
                f"{model_name} -> Train R2: {train_model_score:.4f}, "
                f"Test R2: {test_model_score:.4f}"
            )

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
