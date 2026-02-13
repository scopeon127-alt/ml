# =========================
# Import required libraries
# =========================

import os                      # for file/folder path handling
import sys                     # for exception traceback info
import pandas as pd            # for dataframe operations

from sklearn.model_selection import train_test_split  # for splitting dataset
from dataclasses import dataclass                    # for config class

from src.exception import CustomException            # custom error handling
from src.logger import logging                      # logging system


# ======================================================
# Config class (stores all file paths in one place)
# ======================================================
# @dataclass automatically creates __init__ for us
# Helps keep configuration clean and manageable
# ======================================================

@dataclass
class DataIngestionConfig:

    # Path where training data will be saved
    train_data_path: str = os.path.join('artifacts', "train.csv")

    # Path where testing data will be saved
    test_data_path: str = os.path.join('artifacts', "test.csv")

    # Path where raw original data will be saved
    raw_data_path: str = os.path.join('artifacts', "data.csv")



# ======================================================
# Data Ingestion Component
# Responsibility:
#   1. Read dataset
#   2. Store raw copy
#   3. Split into train/test
#   4. Save files
# ======================================================

class DataIngestion:

    # Constructor
    # Initializes configuration object
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    # Main method to start ingestion process
    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion component")

        try:
            # ==================================================
            # Step 1: Read dataset from notebook/data folder
            # ==================================================
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read successfully as dataframe")


            # ==================================================
            # Step 2: Create artifacts folder if not present
            # ==================================================
            # os.path.dirname gets folder name from file path
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )


            # ==================================================
            # Step 3: Save raw/original data (backup copy)
            # ==================================================
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )


            logging.info("Train-test split initiated")


            # ==================================================
            # Step 4: Split data into train and test
            # 80% train, 20% test
            # random_state ensures same split every run
            # ==================================================
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )


            # ==================================================
            # Step 5: Save train and test datasets separately
            # ==================================================
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )


            logging.info("Data ingestion completed successfully")


            # ==================================================
            # Step 6: Return file paths for next pipeline steps
            # ==================================================
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        # ==================================================
        # Error handling using custom exception
        # ==================================================
        except Exception as e:
            raise CustomException(e, sys)
