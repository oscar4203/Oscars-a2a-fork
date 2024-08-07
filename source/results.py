# Description: Module to log game results and player preferences

# Standard Libraries
import logging
from dataclasses import dataclass
import os
import csv
import numpy as np
from datetime import datetime

# Third-party Libraries

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent, AIAgent, HumanAgent, RandomAgent


# Results Base Directory
RESULTS_BASE_DIRECTORY = "./logs/"


# Game Results Datatype
@dataclass
class GameResults:
    agents: list[Agent]
    points_to_win: int
    round: int
    green_apple: GreenApple
    red_apples: list[RedApple]
    winning_red_apple: RedApple
    winning_player: Agent

    def __post_init__(self) -> None:
        logging.debug(f"Created GameResults object: {self}")

    def __str__(self) -> str:
        return f"GameResults(agents={[player.get_name() for player in self.agents]}, points_to_win={self.points_to_win}, round={self.round}, " \
               f"green_apple={self.green_apple.get_adjective()}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apple={self.winning_red_apple.get_noun()}, winning_player={self.winning_player.get_name()})"

    def __repr__(self) -> str:
        return f"GameResults(agents={[player.get_name() for player in self.agents]}, points_to_win={self.points_to_win}, round={self.round}, " \
               f"green_apple={self.green_apple}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apple={self.winning_red_apple.get_noun()}, winning_player={self.winning_player.get_name()}"

    def to_dict(self) -> dict:
        return {
            "agents": [player.get_name() for player in self.agents],
            "points_to_win": self.points_to_win,
            "round": self.round,
            "green_apple": self.green_apple.get_adjective(),
            "red_apples": [apple.get_noun() for apple in self.red_apples],
            "winning_red_apple": self.winning_red_apple.get_noun(),
            "winning_player": self.winning_player.get_name()
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


def log_preference_updates(preference_updates: PreferenceUpdates) -> None:
    filename = f"{RESULTS_BASE_DIRECTORY}/Game-{preference_updates.time}.csv"

    #Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a') as file:
        file.write("--------------------------------------------\n")
        writer = csv.DictWriter(file, fieldnames=preference_updates.to_dict().keys())
        file_empty = os.path.getsize(filename) == 0
        if file_empty:
            writer.writeheader()
        writer.writerow(preference_updates.to_dict())


def format_players_string(players: list[Agent]) -> str:
    # Initialize the abbreviations lists
    ai_abbrev_list = []
    human_abbrev_list = []
    random_list = []

    # Abbreviate the player names to the first 3 characters
    for player in players:
        if isinstance(player, AIAgent):
            name = player.get_name()
            ai_abbrev_list.append(name[:3])
        elif isinstance(player, HumanAgent):
            human_abbrev_list.append("hum")
        elif isinstance(player, RandomAgent):
            random_list.append("rnd")

    # Join the lists into a single string
    ai_abbrev = "_".join(ai_abbrev_list)

    # Count the number of human players
    human_abbrev = ""
    if len(human_abbrev_list) > 0:
        human_abbrev = f"hum-{len(human_abbrev_list)}"

    # Count the number of random players
    random_abbrev = ""
    if len(random_list) > 0:
        random_abbrev = f"rnd-{len(random_list)}"

    # Join the strings
    return f"{ai_abbrev}-{human_abbrev}-{random_abbrev}"


def format_naming_scheme(players: list[Agent], number_of_games: int | None = None, points_to_win: int | None = None) -> str:
    # Get the current date
    date = datetime.now().strftime("%Y_%m_%d")

    # Format the players string
    players_string = format_players_string(players)

    # Format the naming scheme
    string = f"{date}-"
    if number_of_games is not None:
        string += f"{number_of_games}_games-{players_string}"
    if points_to_win is not None:
        string += f"{points_to_win}_pts-"
    string += f"{players_string}"

    return string


def log_gameplay(game_results: GameResults, number_of_games: int, header: bool) -> None:
    # Define the naming scheme
    naming_scheme = format_naming_scheme(game_results.agents, number_of_games, game_results.points_to_win)

    # Define the directory
    directory = RESULTS_BASE_DIRECTORY + naming_scheme + "/"

    # Define the filename
    filename = "gameplay" + naming_scheme + ".csv"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(directory), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(f"{directory}{filename}", 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=game_results.to_dict().keys())

        # Write a header if requested
        if header:
            # Check if the file is empty
            file_empty = os.path.getsize(f"{directory}{filename}") == 0
            if file_empty:
                writer.writeheader()

        # Write the game results
        writer.writerow(game_results.to_dict())


def log_winner(game_results: GameResults, number_of_games: int, header: bool) -> None:
    # Define the naming scheme
    naming_scheme = format_naming_scheme(game_results.agents, number_of_games, game_results.points_to_win)

    # Define the directory
    directory = RESULTS_BASE_DIRECTORY + naming_scheme + "/"

    # Define the filename
    filename = "winners" + naming_scheme + ".csv"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(directory), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(f"{directory}{filename}", 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=["Winner"])

        # Write a header if requested
        if header:
            # Check if the file is empty
            file_empty = os.path.getsize(f"{directory}{filename}") == 0
            if file_empty:
                writer.writeheader()

        # Write the game results
        writer.writerow({"Winner": game_results.winning_player.get_name()})


def log_training(game_results: GameResults, header: bool) -> None:
    # Agent to be trained
    agents = game_results.agents
    ai_agent = []

    # Find the AI agent
    for agent in agents:
        if isinstance(agent, AIAgent):
            ai_agent.append(agent)
            break

    # Define the naming scheme
    naming_scheme = format_naming_scheme(ai_agent)

    # Define the directory
    directory = RESULTS_BASE_DIRECTORY + naming_scheme + "/"

    # Define the filename
    filename = "training" + naming_scheme + ".csv"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(directory), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(f"{directory}{filename}", 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=game_results.to_dict().keys())

        # Write a header if requested
        if header:
            # Check if the file is empty
            file_empty = os.path.getsize(f"{directory}{filename}") == 0
            if file_empty:
                writer.writeheader()

        # Write the game results
        writer.writerow(game_results.to_dict())


if __name__ == "__main__":
    pass
