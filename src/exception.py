import sys   # used to access system-specific parameters and traceback info


def error_message_detail(error, error_detail: sys):
    """
    This function extracts detailed error information
    like file name and line number where exception occurred.
    """

    # exc_info() returns (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where error happened
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Get the line number where error occurred
    line_number = exc_tb.tb_lineno

    # Create detailed custom error message
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception class that extends Python's default Exception.

    It provides:
    - filename
    - line number
    - actual error message

    Helpful for debugging large ML pipelines.
    """

    def __init__(self, error_message, error_detail: sys):

        # Call parent Exception constructor
        super().__init__(error_message)

        # Store detailed formatted error message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        # When exception is printed, show custom message
        return self.error_message
