# Description: Module to log game results and player preferences

# Standard Libraries
import logging
from dataclasses import dataclass
import os
import csv
import numpy as np

# Third-party Libraries

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent


# Filename constants
GAMEPLAY_FILENAME = "./logs/gameplay.csv"
WINNERS_FILENAME = "./logs/winners.csv"
TRAINING_FILENAME = "./logs/training.csv"
PREFERENCES_FILENAME = "./logs/preferences_round"


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
    filename = f"./logs/Game-{preference_updates.time}.csv"

    #Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'a') as file:
        file.write("--------------------------------------------\n")
        writer = csv.DictWriter(file, fieldnames=preference_updates.to_dict().keys())
        file_empty = os.path.getsize(filename) == 0
        if file_empty:
            writer.writeheader()
        writer.writerow(preference_updates.to_dict())


def log_gameplay(game_results: GameResults, header: bool) -> None:
    # # Check if file exists
    # file_exists = os.path.isfile(RESULTS_FILENAME)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(GAMEPLAY_FILENAME), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(GAMEPLAY_FILENAME, 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=game_results.to_dict().keys())

        # Write a header if requested
        if header:
            # Check if the file is empty
            file_empty = os.path.getsize(WINNERS_FILENAME) == 0
            if file_empty:
                writer.writeheader()

        # Write the game results
        writer.writerow(game_results.to_dict())


def log_winner(winner: Agent, header: bool) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(WINNERS_FILENAME), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(WINNERS_FILENAME, 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=["Winner"])

        # Write a header if requested
        if header:
            # Check if the file is empty
            file_empty = os.path.getsize(WINNERS_FILENAME) == 0
            if file_empty:
                writer.writeheader()

        # Write the game results
        writer.writerow({"Winner": winner.get_name()})


def log_training(game_results: GameResults, header: bool) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(TRAINING_FILENAME), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(TRAINING_FILENAME, 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=game_results.to_dict().keys())

        # Write a header if requested
        if header:
            # Check if the file is empty
            file_empty = os.path.getsize(WINNERS_FILENAME) == 0
            if file_empty:
                writer.writeheader()

        # Write the game results
        writer.writerow(game_results.to_dict())


if __name__ == "__main__":
    pass
