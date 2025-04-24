# Description: Module to log game results and player preferences

# Standard Libraries
from typing import TYPE_CHECKING # Type hinting to avoid circular imports
import os
import csv
import logging
from datetime import datetime
import numpy as np

# Third-party Libraries

# Local Modules
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.data_classes.data_classes import GameState, GameLog
from src.data_classes.data_classes import PathsConfig


# Logging configuration
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s (%(name)s) %(module)s.%(funcName)s:%(lineno)d - %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(debug_mode: bool, paths_config: PathsConfig) -> None:
    """
    Configure logging parameters for the application.
    """
    try: # Add try block for better error visibility
        log_dir = paths_config.logging_base_directory
        log_filename = paths_config.logging_filename
        file_path = os.path.join(log_dir, log_filename)

        if debug_mode:
            print(f"[DEBUG] configure_logging: Attempting to use log file path: {os.path.abspath(file_path)}") # Show absolute path

        # Check that the logging directory can be created
        # os.makedirs will raise an error if it fails and exist_ok is False,
        # but with exist_ok=True, we might want to check writability if needed.
        # For now, let's assume makedirs works or the dir exists.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if debug_mode:
            print(f"[DEBUG] configure_logging: Ensured directory exists: {os.path.abspath(os.path.dirname(file_path))}")

        # Set the logging level
        logging_level = logging.DEBUG if debug_mode else logging.INFO
        if debug_mode:
            print(f"[DEBUG] configure_logging: Setting logging level to: {logging_level} ({'DEBUG' if debug_mode else 'INFO'})")

        # Configure logging
        logging.basicConfig(
            level=logging_level,
            format=LOGGING_FORMAT,
            datefmt=LOGGING_DATE_FORMAT,
            filename=file_path,
            filemode='w' # Force overwrite mode for debugging
        )
        if debug_mode:
            print(f"[DEBUG] configure_logging: logging.basicConfig called for file: {os.path.abspath(file_path)}")

    except Exception as e:
        if debug_mode:
            print(f"[ERROR] configure_logging: Failed during logging setup: {e}")
        # Optionally re-raise or handle differently
        raise


def log_to_csv(directory: str, filename: str, fieldnames: list[str], data: dict, header: bool = True) -> None:
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(file_path, 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write a header if requested and the file is empty
        if header and os.path.getsize(file_path) == 0:
            writer.writeheader()

        # Write the data
        writer.writerow(data)


def log_vectors(paths_config: PathsConfig, game_log: "GameLog", game_state: "GameState", judge: "Agent", player: "Agent", slope: np.ndarray, bias: np.ndarray, header: bool) -> None:
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    directory = os.path.join(paths_config.logging_base_directory, game_log.naming_scheme)
    filename = f"vectors-{game_log.naming_scheme}.csv"

    # Runtime import to avoid circular dependency
    from src.data_classes.data_classes import PreferenceUpdates

    preference_updates = PreferenceUpdates(judge, player, game_log.get_current_game_number(), game_log.get_current_round_number(), date_time,
                                           game_state.get_current_round_chosen_apples().get_green_apple(),
                                           game_state.get_current_round_chosen_apples().get_winning_red_apple(),
                                           slope, bias)
    log_to_csv(directory, filename, list(preference_updates.to_dict().keys()), preference_updates.to_dict(), header)


def log_game_state(paths_config: PathsConfig, game_log: "GameLog", header: bool) -> None:
    directory = os.path.join(paths_config.logging_base_directory, game_log.naming_scheme)
    filename = f"game_state-{game_log.naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_log.game_log_to_dict().keys()), game_log.game_log_to_dict(), header)


def log_round_winner(paths_config: PathsConfig, game_log: "GameLog", game_state: "GameState", header: bool) -> None:
    directory = os.path.join(paths_config.logging_base_directory, game_log.naming_scheme)
    filename = f"round_winners-{game_log.naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.round_winner_to_dict().keys()), game_state.round_winner_to_dict(), header)


def log_game_winner(paths_config: PathsConfig, game_log: "GameLog", game_state: "GameState", header: bool) -> None:
    directory = os.path.join(paths_config.logging_base_directory, game_log.naming_scheme)
    filename = f"game_winners-{game_log.naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.game_winner_to_dict().keys()), game_state.game_winner_to_dict(), header)


def log_training_mode(paths_config: PathsConfig, game_log: "GameLog", game_state: "GameState", header: bool) -> None:
    directory = os.path.join(paths_config.logging_base_directory, game_log.naming_scheme)
    filename = f"training-{game_log.naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.training_to_dict().keys()), game_state.training_to_dict(), header)


if __name__ == "__main__":
    pass
