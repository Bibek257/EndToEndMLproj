import sys
import types
import warnings
import logging

from logger import logging  
# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def error_message_details(error: Exception, error_details: types.ModuleType) -> str:
    """
    Returns a formatted error message with file name and line number.
    """
    _, _, tb = sys.exc_info()
    if tb is None:
        return f"An unknown error occurred: {str(error)}"

    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    return (
        f"Error occurred in script [{file_name}] "
        f"at line number [{line_number}] "
        f"with message [{str(error)}]"
    )


class CustomException(Exception):
    """
    Custom exception class that logs the error when raised.
    """

    def __init__(self, error: Exception, error_details: types.ModuleType):
        super().__init__(str(error))
        self.error_message = error_message_details(error, error_details)

        # Log the error automatically
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    # Example fallback logging config (if logger.py isn't loaded)
    logging.basicConfig(
        filename="error.log",
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    )

    try:
        a = 1 / 0  # Intentional error for testing
    except Exception as e:
        raise CustomException(e, sys)
