import sys       # gives access to system-level info like traceback
import src.logger
import logging  # used to record logs


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information such as:
    - file name
    - line number
    - actual error message
    """

    # exc_info() returns: (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()

    # File where exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Line number of exception
    line_number = exc_tb.tb_lineno

    # Create readable custom message
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception class for better debugging.

    Instead of normal error:
        ZeroDivisionError: division by zero

    We get:
        file name + line number + message
    """

    def __init__(self, error_message, error_detail: sys):
        # Call parent constructor
        super().__init__(error_message)

        # Save detailed message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# ===============================
# Testing the custom exception
# ===============================
if __name__ == "__main__":
    try:
        a = 2 / 0   # will cause ZeroDivisionError

    except Exception as e:
        logging.info("Divide by zero error")

        # must pass error + sys
        raise CustomException(e, sys)
