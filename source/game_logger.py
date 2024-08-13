# Description: Module to log game results and player preferences

# Standard Libraries
from typing import TYPE_CHECKING # Type hinting to avoid circular imports
import os
import csv
import logging
from datetime import datetime

# Third-party Libraries

# Local Modules
# if TYPE_CHECKING:
from source.agent import Agent, AIAgent, HumanAgent, RandomAgent
from source.data_classes import GameState, PreferenceUpdates


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


def format_players_string(players: list[Agent]) -> str:
    # Initialize the abbreviations lists
    ai_abbrev_list = []
    human_abbrev_list = []
    random_list = []

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


def format_naming_scheme(players: list[Agent], total_games: int | None = None, points_to_win: int | None = None) -> str:
    # Get the current date
    date = datetime.now().strftime("%Y_%m_%d")

    # Format the players string
    players_string = format_players_string(players)

    # Format the naming scheme
    string = f"{date}-"
    if total_games is not None:
        string += f"{total_games}_games-"
    if points_to_win is not None:
        string += f"{points_to_win}_pts-"
    string += f"{players_string}"

    return string


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


# def log_vectors(game_state: GameState, player: Agent, current_slope, current_bias, header: bool) -> None:
#     date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
#     naming_scheme = format_naming_scheme(game_state.players, game_state.total_games, game_state.points_to_win)
#     directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
#     filename = f"vectors-{naming_scheme}.csv"
#     # Check that the green and red apples and current judge are not None
#     if game_state.green_apple is None or game_state.winning_red_apple is None or game_state.current_judge is None:
#         raise ValueError("Green apple or winning red apple or current judge is None.")
#     preference_updates = PreferenceUpdates(player, game_state.current_round, date_time,
#                                            game_state.green_apple[game_state.current_judge], game_state.winning_red_apple,
#                                            current_slope, current_bias)
#     log_to_csv(directory, filename, list(preference_updates.to_dict().keys()), preference_updates.to_dict(), header)


def log_gameplay(game_state: GameState, header: bool) -> None:
    naming_scheme = format_naming_scheme(game_state.players, game_state.total_games, game_state.points_to_win)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"gameplay-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.gameplay_to_dict().keys()), game_state.gameplay_to_dict(), header)


def log_winner(game_state: GameState, header: bool) -> None:
    naming_scheme = format_naming_scheme(game_state.players, game_state.total_games, game_state.points_to_win)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"winners-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.winner_to_dict().keys()), game_state.winner_to_dict(), header)


def log_training(game_state: GameState, header: bool) -> None:
    naming_scheme = format_naming_scheme(game_state.players, game_state.total_games, game_state.points_to_win)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"training-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_state.training_to_dict().keys()), game_state.training_to_dict(), header)


if __name__ == "__main__":
    pass
