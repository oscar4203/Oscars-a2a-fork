# Description: Configuration file for the application

# Standard Libraries
import logging

# Third-party Libraries

# Local Modules
from source.config import configure_logging
from source.results import GameResults, log_results


def game_loop() -> None:
    logging.info("Starting 'Apples to Apples' game loop.")


def main() -> None:
    # Configure logging
    configure_logging()




if __name__ == "__main__":
    main()
