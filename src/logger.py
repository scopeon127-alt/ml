import logging
import os
from datetime import datetime

"""
logger.py

This file configures logging for the entire project.
It should be imported ONLY ONCE at application start.

Features:
✔ Creates logs folder automatically
✔ Creates timestamped log file
✔ Logs to file + console
✔ Used across whole ML pipeline
"""


# =========================
# Create logs directory
# =========================

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# =========================
# Create unique log file
# =========================

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)


# =========================
# Configure logging
# =========================

logging.basicConfig(
    level=logging.INFO,

    # log message format
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # log to BOTH file and console
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)


# =========================
# Test run (optional)
# =========================
if __name__ == "__main__":
    logging.info("Logger initialized successfully")
