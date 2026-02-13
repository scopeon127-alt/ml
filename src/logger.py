import logging
import os
from datetime import datetime

# Create log file name using current date and time
# Example: 02_13_2026_14_30_55.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs directory path
# Example: project/logs
logs_dir = os.path.join(os.getcwd(), "logs")

# Create logs folder if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Final full log file path
# Example: project/logs/02_13_2026_14_30_55.log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)


# Configure logging settings
logging.basicConfig(

    # Where logs will be stored
    filename=LOG_FILE_PATH,

    # Log format
    # asctime -> time
    # lineno  -> line number
    # name    -> module name
    # level   -> INFO/ERROR/WARNING
    # message -> actual log message
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # Minimum level to log
    level=logging.INFO,
)
