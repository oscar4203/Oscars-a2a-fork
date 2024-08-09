# Description: Module to log game results and player preferences

# Standard Libraries
import os
import csv
import logging
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Third-party Libraries

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent, AIAgent, HumanAgent, RandomAgent


# Logging configuration
DEBUG_MODE = True
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s (%(name)s) %(module)s - %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_BASE_DIRECTORY = "./logs/"
LOGGING_FILENAME = "apples_to_apples.log"


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
    file_path = os.path.join(LOGGING_BASE_DIRECTORY, LOGGING_FILENAME)

    # Check that the logging file and directory exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        filename=file_path
    )


# Game Results Datatype
@dataclass
class GameResults:
    agents: list[Agent]
    points_to_win: int
    total_games: int
    current_game: int
    round: int
    green_apple: GreenApple
    red_apples: list[RedApple]
    winning_red_apple: RedApple
    losing_red_apples: list[RedApple]
    current_judge: Agent
    round_winner: Agent | None = None
    game_winner: Agent | None = None

    def __post_init__(self) -> None:
        logging.debug(f"Created GameResults object: {self}")

    def __str__(self) -> str:
        return f"GameResults(agents={[player.get_name() for player in self.agents]}, "\
               f"points_to_win={self.points_to_win}, total_games={self.total_games}, "\
               f"current_game={self.current_game}, round={self.round}, "\
               f"green_apple={self.green_apple.get_adjective()}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apple={self.winning_red_apple.get_noun()}, "\
               f"losing_red_apples={[apple.get_noun() for apple in self.losing_red_apples]}, "\
               f"current_judge={self.current_judge.get_name()}, "\
               f"round_winner={self.round_winner.get_name() if self.round_winner is not None else None}, "\
               f"game_winner={self.game_winner.get_name() if self.game_winner is not None else None})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict[str, list[str] | str | int | None]:
        return {
            "agents": [player.get_name() for player in self.agents],
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "round": self.round,
            "green_apple": self.green_apple.get_adjective(),
            "red_apples": [apple.get_noun() for apple in self.red_apples],
            "winning_red_apple": self.winning_red_apple.get_noun(),
            "losing_red_apples": [apple.get_noun() for apple in self.losing_red_apples],
            "current_judge": self.current_judge.get_name(),
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None,
            "game_winner": self.game_winner.get_name() if self.game_winner is not None else None
        }


@dataclass
class PreferenceUpdates:
    agent: Agent
    round: int
    time: str
    winning_red_apple: RedApple
    green_apple: GreenApple
    bias: np.ndarray
    slope: np.ndarray

    def __str__(self) -> str:
        return f"PreferenceUpdates(agent={self.agent.get_name()}, round={self.round}, time={self.time}, winning red apple={self.winning_red_apple.get_noun()}, green apple={self.green_apple.get_adjective()}, bias={self.bias}, slope={self.slope})"
    def __repr__(self) -> str:
         return f"PreferenceUpdates(agent={self.agent.get_name()}, round={self.round}, time={self.time}, winning red apple={self.winning_red_apple.get_noun()}, green apple={self.green_apple.get_adjective()}, bias={self.bias}, slope={self.slope})"

    def to_dict(self) -> dict:
        return {
            "Agent": self.agent.get_name(),
            "round": self.round,
            "time": self.time,
            "green_apple": self.green_apple.get_adjective(),
            "winning_red_apple": self.winning_red_apple.get_noun(),
            "Bias": f"{self.bias}\n",
            "Slope": f"{self.slope}\n"
        }


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


def log_to_csv(directory: str, filename: str, fieldnames: list[str], data: dict, header: bool) -> None:
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


def log_vectors(game_results: GameResults, preference_updates: PreferenceUpdates) -> None:
    naming_scheme = format_naming_scheme(game_results.agents, game_results.total_games, game_results.points_to_win)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"vectors-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(preference_updates.to_dict().keys()), preference_updates.to_dict(), header=True)


def log_gameplay(game_results: GameResults,header: bool) -> None:
    naming_scheme = format_naming_scheme(game_results.agents, game_results.total_games, game_results.points_to_win)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"gameplay-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_results.to_dict().keys()), game_results.to_dict(), header)


def log_winner(game_results: GameResults, header: bool) -> None:
    naming_scheme = format_naming_scheme(game_results.agents, game_results.total_games, game_results.points_to_win)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"winners-{naming_scheme}.csv"
    log_to_csv(directory, filename, ["Game Winner"],
               {"Game Winner": game_results.game_winner.get_name()} \
               if game_results.game_winner is not None else {"Game Winner": None}, header)


def log_training(game_results: GameResults, header: bool) -> None:
    agents: list[Agent] = game_results.agents
    ai_agent: list[Agent] = [agent for agent in agents if isinstance(agent, AIAgent)]
    naming_scheme = format_naming_scheme(ai_agent)
    directory = os.path.join(LOGGING_BASE_DIRECTORY, naming_scheme)
    filename = f"training-{naming_scheme}.csv"
    log_to_csv(directory, filename, list(game_results.to_dict().keys()), game_results.to_dict(), header)


if __name__ == "__main__":
    pass
