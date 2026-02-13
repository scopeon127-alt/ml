# =========================================================
# Imports
# =========================================================

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object   # utility function to save pickle


# =========================================================
# Configuration Class
# Stores file paths used in transformation
# =========================================================

@dataclass
class DataTransformationConfig:

    # Path where preprocessing object will be saved
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
#   5. Return processed arrays
# =========================================================

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    # =====================================================
    # Create preprocessing object (ColumnTransformer)
    # =====================================================
    def get_data_transformer_object(self):

        try:
            logging.info("Creating preprocessing pipelines")


            # =============================================
            # Separate column types
            # =============================================

            # Numerical features → scaling needed
            numerical_columns = [
                "writing score",
                "reading score"
            ]

            # Categorical features → encoding needed
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]


            # =============================================
            # Numerical Pipeline
            #   1. Fill missing with median
            #   2. Scale values
            # =============================================
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )


            # =============================================
            # Categorical Pipeline
            #   1. Fill missing with most frequent
            #   2. One hot encoding
            # =============================================
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )


            # =============================================
            # Combine both pipelines
            # =============================================
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
    # Main transformation method
    # =====================================================
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading train and test datasets")


            # =============================================
            # Read datasets
            # =============================================
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            # =============================================
            # Get preprocessing object
            # =============================================
            preprocessing_obj = self.get_data_transformer_object()


            # =============================================
            # Target column
            # =============================================
            target_column_name = "math score"


            # =============================================
            # Separate input and target
            # =============================================
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]


            logging.info("Applying preprocessing")


            # ==================================================
            # VERY IMPORTANT (ML best practice)
            # Fit ONLY on train
            # Transform test
            # ==================================================
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )


            # =============================================
            # Combine features + target
            # =============================================
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]


            # =============================================
            # Save preprocessing object for future prediction
            # =============================================
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully")


            # =============================================
            # Return arrays and file path
            # =============================================
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
