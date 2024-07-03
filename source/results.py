# Description: Configuration file for the application

# Standard Libraries
import logging
from dataclasses import dataclass
import os
import csv

# Third-party Libraries

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent

# Results constants
RESULTS_FILENAME = "./logs/results.csv"


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
        return f"GameResults(agents={[player.name for player in self.agents]}, points_to_win={self.points_to_win}, round={self.round}, " \
               f"green_apple={self.green_apple.adjective}, red_apples={[apple.noun for apple in self.red_apples]}, " \
               f"winning_red_apple={self.winning_red_apple.noun}, winning_player={self.winning_player.name})"

    def __repr__(self) -> str:
        return f"GameResults(agents={[player.name for player in self.agents]}, points_to_win={self.points_to_win}, round={self.round}, " \
               f"green_apple={self.green_apple}, red_apples={[apple.noun for apple in self.red_apples]}, " \
               f"winning_red_apple={self.winning_red_apple.noun}, winning_player={self.winning_player.name}"

    def to_dict(self) -> dict:
        return {
            "agents": [player.name for player in self.agents],
            "points_to_win": self.points_to_win,
            "round": self.round,
            "green_apple": self.green_apple.adjective,
            "red_apples": [apple.noun for apple in self.red_apples],
            "winning_red_apple": self.winning_red_apple.noun,
            "winning_player": self.winning_player.name
        }


def log_results(game_results: GameResults) -> None:
    # # Check if file exists
    # file_exists = os.path.isfile(RESULTS_FILENAME)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(RESULTS_FILENAME), exist_ok=True)

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(RESULTS_FILENAME, 'a') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=game_results.to_dict().keys())

        # Check if the file is empty
        file_empty = os.path.getsize(RESULTS_FILENAME) == 0
        if file_empty:
            writer.writeheader()

        # Write the game results
        writer.writerow(game_results.to_dict())


if __name__ == "__main__":
    pass
