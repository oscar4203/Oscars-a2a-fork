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

#Maybe bad because of circular importing?
import numpy as np

# Results constants
RESULTS_FILENAME = "./logs/results.csv"
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

@dataclass 
class JudgePreferences:
    agent: Agent
    round: int
    biases: np.ndarray
    slope: np.ndarray

    def __str__(self) -> str:
        return {f"JudgePreferences(agent={self.agent.name}, round={self.round}, biases={self.biases}, slopes={self.slope})"}
    def __repr__(self) -> str:
        return {f"JudgePreferences(agent={self.agent.name}, round={self.round}, biases={self.biases}, slopes={self.slope})"}

    def to_dict(self) -> dict:
        return {
            "Round": self.round,
            "Biases": self.biases,
            "Slopes": self.slope
        }
    
@dataclass 
class PreferenceUpdates:
    agents: list[Agent]
    round: int
    time: str
    # judge: Agent
    winning_red_apple: RedApple
    green_apple: GreenApple
    biases_list: list[np.ndarray]
    slopes_list: list[np.ndarray]

    def __str__(self) -> str:
        agent_str = "Agent's Biases' and Slopes:\n"
        for player in self.agents:
            agent_str += f"{player.name}: Bias:\n{self.biases_list[player]}\nSlope:\n{self.slope_list[player]}\n"
        return {f"-------------Round: {self.round}, Time: {self.time}, "
                f"winning red: {self.winning_red.name}, Green: {self.green_apple.name}--------\n"
                f"{agent_str}"}
    def __repr__(self) -> str:
        return {f"PreferenceUpdates(agents={[player.name for player in self.agents]}"}

    def to_dict(self) -> dict:
        return {
            "agents": self.agents,
            "round": self.round,
            "time": self.time,
            "green_apple": self.green_apple.adjective,
            "winning_red_apple": self.winning_red_apple.noun,
            #"Judge":
        }

def log_preference_updates(preference_updates: PreferenceUpdates) -> None:
    filename = f"./logs/Game-{preference_updates.time}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as file:
        writer = csv.DictWriter(file, fieldnames=preference_updates.to_dict().keys())
        file_empty = os.path.getsize(filename) == 0
        if file_empty:
            writer.writeheader()
        writer.writerow(preference_updates.to_dict())

def log_preferences(judge_preferences: JudgePreferences) -> None:
    filename = f"{PREFERENCES_FILENAME}{judge_preferences.round}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as file:
        writer = csv.DictWriter(file, fieldnames=judge_preferences.to_dict().keys())
        file_empty = os.path.getsize(filename) == 0
        if file_empty:
            writer.writeheader()
        writer.writerow(judge_preferences.to_dict())

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
