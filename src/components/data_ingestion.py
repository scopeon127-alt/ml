# ======================================================
# Import required libraries
# ======================================================

import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# Import pipeline components
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



# ======================================================
# Config class
# ======================================================
@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")



# ======================================================
# Data Ingestion Component
# ======================================================
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):

        logging.info("Entered Data Ingestion component")

        try:
            # Step 1: Read dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset loaded successfully")

            # Step 2: Create artifacts folder
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Step 3: Save raw copy
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Step 4: Train-Test Split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Step 5: Save split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)



# ======================================================
# Pipeline Runner (Entry Point)
# ======================================================
if __name__ == "__main__":

    try:
        logging.info("Pipeline started")

        # --------------------------------------
        # Step 1: Data Ingestion
        # --------------------------------------
        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()


        # --------------------------------------
        # Step 2: Data Transformation
        # --------------------------------------
        transformation_obj = DataTransformation()
        train_arr, test_arr, _ = transformation_obj.initiate_data_transformation(
            train_path, test_path
        )


        # --------------------------------------
        # Step 3: Model Training
        # --------------------------------------
        model_trainer = ModelTrainer()
        r2_score_value = model_trainer.initiate_model_trainer(
            train_arr, test_arr
        )

        print(f"\nFinal R2 Score: {r2_score_value:.4f}")

        logging.info("Pipeline completed successfully")

    except Exception as e:
        raise CustomException(e, sys)
