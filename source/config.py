# Description: Configuration file for the application

# Standard Libraries
import logging

# Third-party Libraries

# Local Modules


# Logging configuration
DEBUG_MODE = True
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s (%(name)s) %(module)s - %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FILENAME = "../logs/apples_to_apples.log"


def configure_logging() -> None:
    """
    Configure logging parameters for the application.

    Example usage:
    ```python
    def main() -> None:
        # Configure and initialize the logging module
        configure_logging()
    ```
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else LOGGING_LEVEL,
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        filename=LOGGING_FILENAME
    )


if __name__ == "__main__":
    pass
