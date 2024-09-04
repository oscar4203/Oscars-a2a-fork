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
    from source.agent import Agent
    from source.data_classes import GameState, GameLog


# Logging configuration
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s (%(name)s) %(module)s.%(funcName)s:%(lineno)d - %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_BASE_DIRECTORY = "./logs/"
LOGGING_FILENAME = "apples_to_apples.log"


def configure_logging(debug_mode: bool) -> None:
    """
    Configure logging parameters for the application.

    Example usage:
    ```python
    def main() -> None:
        # Configure and initialize the logging module
        configure_logging()
    ```
    """
    file_path = os.path.join(LOGGING_BASE_DIRECTORY, LOGGING_FILENAME)

    # Check that the logging file and directory exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        filename=file_path
    )


def format_players_string(players: "list[Agent]") -> str:
    # Initialize the abbreviations lists
    ai_abbrev_list = []
    human_abbrev_list = []
    random_list = []

    # Runtime import to avoid circular dependency
    from source.agent import AIAgent, HumanAgent, RandomAgent

    # Abbreviate the player names to the first 3 characters
    for player in players:
        if isinstance(player, AIAgent):
            name: str = player.get_name().lower()
            name_parts: list[str] = name.split("-")
            last_part: str = name_parts[-1].strip()
            ai_abbrev_list.append(last_part[:3])
        elif isinstance(player, HumanAgent):
            human_abbrev_list.append("hum")
        elif isinstance(player, RandomAgent):
            random_list.append("rnd")

    # Join the lists into a single string
    ai_abbrev = "_".join(ai_abbrev_list)

    # Count the number of human players
    human_abbrev = ""
    if len(human_abbrev_list) > 0:
        human_abbrev = f"hum_{len(human_abbrev_list)}"

    # Count the number of random players
    random_abbrev = ""
    if len(random_list) > 0:
        random_abbrev = f"rnd_{len(random_list)}"

    # Join the strings
    final_string: str = "-".join(filter(None, [ai_abbrev, human_abbrev, random_abbrev]))
    return final_string


def create_final_naming_scheme_string(date_string: str, total_games: int | None = None, points_to_win: int | None = None, players_string: str = "", num: int | None = None) -> str:
    num_string = f"{num}-" if num else ""
    return f"{date_string}{num_string}{total_games}_games-{points_to_win}_pts-{players_string}"


def format_naming_scheme(players: "list[Agent]", total_games: int | None = None, points_to_win: int | None = None) -> str:
    # Get the current date
    date = datetime.now().strftime("%Y_%m_%d")

    # Format the players string
    players_string = format_players_string(players)

    # Format the initial naming scheme
    date_string = f"{date}-"
    num = 1
    final_string = create_final_naming_scheme_string(date_string, total_games, points_to_win, players_string, num)

    # Check if there already exists a directory with the same naming scheme
    while os.path.exists(os.path.join(LOGGING_BASE_DIRECTORY, final_string)):
        num += 1
        final_string = create_final_naming_scheme_string(date_string, total_games, points_to_win, players_string, num)

    return final_string


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


def log_vectors(game_log: "GameLog", game_state: "GameState", judge: "Agent", player: "Agent", slope: np.ndarray, bias: np.ndarray, header: bool) -> None:
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    naming_scheme = game_log.naming_scheme
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"vectors-{naming_scheme}.csv"

    # Runtime import to avoid circular dependency
    from source.data_classes import PreferenceUpdates

    preference_updates = PreferenceUpdates(judge, player, game_log.get_current_game_number(), game_log.get_current_round_number(), date_time,
                                           game_state.get_current_round_chosen_apples().get_green_apple(),
                                           game_state.get_current_round_chosen_apples().get_winning_red_apple(),
                                           slope, bias)
    log_to_csv(directory, filename, list(preference_updates.to_dict().keys()), preference_updates.to_dict(), header)


def log_game_state(naming_scheme: str, game_log: "GameLog", header: bool) -> None:
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"game_state-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_log.game_log_to_dict().keys()), game_log.game_log_to_dict(), header)


def log_round_winner(naming_scheme: str, game_state: "GameState", header: bool) -> None:
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"round_winners-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.round_winner_to_dict().keys()), game_state.round_winner_to_dict(), header)


def log_game_winner(naming_scheme: str, game_state: "GameState", header: bool) -> None:
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"game_winners-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.game_winner_to_dict().keys()), game_state.game_winner_to_dict(), header)


def log_training_mode(naming_scheme: str, game_state: "GameState", header: bool) -> None:
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"training-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.training_to_dict().keys()), game_state.training_to_dict(), header)


if __name__ == "__main__":
    pass
