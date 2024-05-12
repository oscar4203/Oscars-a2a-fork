# Description: Configuration file for the application

# Standard Libraries
import logging
from dataclasses import dataclass
import os
import csv

# Third-party Libraries

# Local Modules


# Results constants
RESULTS_FILENAME = "./logs/results.csv"


# Game Results Datatype
@dataclass
class GameResults:
    players: list[str]
    points_to_win: int
    round: bool
    green_card: str
    red_cards: list[str]
    winning_red_card: str
    winning_player: str

    def __post_init__(self) -> None:
        logging.debug(f"Created GameResults object: {self}")

    def __str__(self) -> str:
        return f"GameResults(players={self.players}, points_to_win={self.points_to_win}, round={self.round}, " \
               f"green_card={self.green_card}, red_cards={self.red_cards}, " \
               f"winning_red_card={self.winning_red_card}, winning_player={self.winning_player})"

    def to_dict(self) -> dict:
        return {
            "players": self.players,
            "points_to_win": self.points_to_win,
            "round": self.round,
            "green_card": self.green_card,
            "red_cards": self.red_cards,
            "winning_red_card": self.winning_red_card,
            "winning_player": self.winning_player
        }


def log_results(game_results: GameResults) -> None:
    # Check if file exists and if it's empty
    file_exists = os.path.isfile(RESULTS_FILENAME)
    file_empty = os.path.getsize(RESULTS_FILENAME) == 0

    # Open the file in append mode. This will create the file if it doesn't exist
    with open(RESULTS_FILENAME, 'a') as file:
        writer = csv.DictWriter(file, fieldnames=game_results.to_dict().keys())

        # If file didn't exist or was empty, write the header
        if not file_exists or file_empty:
            writer.writeheader()

        # Write the game results
        writer.writerow(game_results.to_dict())


if __name__ == "__main__":
    pass
