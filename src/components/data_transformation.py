# =========================================================
# Imports
# =========================================================

import os                      # for path handling
import sys                     # for traceback info in exceptions
from dataclasses import dataclass

import numpy as np
import pandas as pd

# sklearn preprocessing tools
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# custom project modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object   # function to save pickle files


# =========================================================
# Configuration Class
# Purpose:
#   Stores all file paths related to transformation
#   (keeps code clean and configurable)
# =========================================================

@dataclass
class DataTransformationConfig:

    # Location where preprocessing object will be saved
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# =========================================================
# Data Transformation Component
# Responsibilities:
#   1. Handle missing values
#   2. Scale numerical features
#   3. Encode categorical features
#   4. Save preprocessing object
#   5. Return transformed arrays
# =========================================================

class DataTransformation:

    def __init__(self):
        # Load configuration
        self.data_transformation_config = DataTransformationConfig()


    # =====================================================
    # Create preprocessing object
    # This object defines HOW data will be transformed
    # =====================================================
    def get_data_transformer_object(self):

        try:
            logging.info("Creating preprocessing pipelines")


            # -------------------------------------------------
            # Separate column types
            # -------------------------------------------------

            # Numerical columns → scaling required
            numerical_columns = [
                "writing score",
                "reading score"
            ]

            # Categorical columns → encoding required
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]


            # -------------------------------------------------
            # Numerical Pipeline
            # Steps:
            #   1. Replace missing values using median
            #   2. Standard scaling (mean=0, std=1)
            # -------------------------------------------------
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )


            # -------------------------------------------------
            # Categorical Pipeline
            # Steps:
            #   1. Replace missing using most frequent
            #   2. Convert categories → numbers (OneHot)
            # -------------------------------------------------
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )


            # -------------------------------------------------
            # ColumnTransformer
            # Applies different pipelines to different columns
            # -------------------------------------------------
            preprocessing = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing object created successfully")

            return preprocessing


        except Exception as e:
            raise CustomException(e, sys)


    # =====================================================
    # Main Transformation Function
    # Flow:
    #   Read data → preprocess → save object → return arrays
    # =====================================================
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading train and test datasets")


            # -------------------------------------------------
            # Step 1: Load datasets
            # -------------------------------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            # -------------------------------------------------
            # Step 2: Get preprocessing object
            # -------------------------------------------------
            preprocessing_obj = self.get_data_transformer_object()


            # -------------------------------------------------
            # Step 3: Define target column
            # -------------------------------------------------
            target_column_name = "math score"


            # -------------------------------------------------
            # Step 4: Split input features and target
            # -------------------------------------------------
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]


            logging.info("Applying preprocessing")


            # ==================================================
            # VERY IMPORTANT (ML best practice)
            #
            # fit() ONLY on training data
            # transform() on test data
            #
            # Prevents data leakage
            # ==================================================
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )


            # -------------------------------------------------
            # Step 5: Combine features + target
            # np.c_ concatenates horizontally
            # -------------------------------------------------
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]


            # -------------------------------------------------
            # Step 6: Save preprocessing object
            # Needed later for prediction/inference
            # -------------------------------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully")


            # -------------------------------------------------
            # Step 7: Return processed arrays
            # -------------------------------------------------
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
