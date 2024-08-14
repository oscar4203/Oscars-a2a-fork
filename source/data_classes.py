# Description: Data classes for the game state and preference updates.

# Standard Libraries
from typing import TYPE_CHECKING # Type hinting to avoid circular imports
import logging
import numpy as np
from dataclasses import dataclass

# Third-party Libraries

# Local Modules
if TYPE_CHECKING:
    from source.apples import GreenApple, RedApple
    from source.agent import Agent


@dataclass
class ApplesInPlay:
    green_apple: dict["Agent", "GreenApple"] | None
    red_apples: list[dict["Agent", "RedApple"]]

    def __post_init__(self) -> None:
        logging.debug(f"Created ApplesInPlay object: {self}")

    def __str__(self) -> str:
        return f"ApplesInPlay(green_apple={self.green_apple}, red_apples={self.red_apples})"

    def to_dict(self) -> dict[str, dict[str, str] | list[dict[str, str]] | None]:
        return {
            "Green Apple": {agent.get_name(): green_apple.get_adjective() for agent, green_apple in self.green_apple.items()} if self.green_apple is not None else None,
            "Red Apples": [{agent.get_name(): red_apple.get_noun() for agent, red_apple in red_apple_dict.items()} for red_apple_dict in self.red_apples]
        }

    def get_green_apple(self) -> "GreenApple":
        if self.green_apple is None:
            raise ValueError("Green apple is not in play.")
        return list(self.green_apple.values())[0]

    def get_red_apples(self) -> list["RedApple"]:
        return list(self.red_apples[0].values())


@dataclass
class ChosenApples:
    green_apple: dict["Agent", "GreenApple"] | None
    winning_red_apple: dict["Agent", "RedApple"] | None
    losing_red_apples: list[dict["Agent", "RedApple"]]

    def __post_init__(self) -> None:
        logging.debug(f"Created ChosenApples object: {self}")

    def __str__(self) -> str:
        return f"ChosenApples(green_apple={self.green_apple}, winning_red_apple={self.winning_red_apple}, losing_red_apples={self.losing_red_apples})"

    def to_dict(self) -> dict[str, dict[str, str] | list[dict[str, str]] | None]:
        return {
            "Green Apple": {agent.get_name(): green_apple.get_adjective() for agent, green_apple in self.green_apple.items()} if self.green_apple is not None else None,
            "Winning Red Apple": {agent.get_name(): red_apple.get_noun() for agent, red_apple in self.winning_red_apple.items()} if self.winning_red_apple is not None else None,
            "Losing Red Apples": [{agent.get_name(): red_apple.get_noun() for agent, red_apple in red_apple_dict.items()} for red_apple_dict in self.losing_red_apples]
        }

    def get_green_apple(self) -> "GreenApple":
        if self.green_apple is None:
            raise ValueError("Green apple has not been drawn yet.")
        return list(self.green_apple.values())[0]

    def get_winning_red_apple(self) -> "RedApple":
        if self.winning_red_apple is None:
            raise ValueError("Winning red apple has not been picked yet.")
        return list(self.winning_red_apple.values())[0]

    def get_losing_red_apples(self) -> list["RedApple"]:
        red_apple_list: list["RedApple"] = []
        for losing_red_apple_dict in self.losing_red_apples:
            for red_apple in losing_red_apple_dict.values():
                red_apple_list.append(red_apple)
        return red_apple_list

    def get_red_apple_winner(self) -> "Agent":
        if self.winning_red_apple is None:
            raise ValueError("Winning red apple has not been picked yet.")
        return list(self.winning_red_apple.keys())[0]


@dataclass
class ChosenAppleVectors:
    green_apple_vector: np.ndarray
    winning_red_apple_vector: np.ndarray
    losing_red_apple_vectors: np.ndarray

    def __post_init__(self) -> None:
        logging.debug(f"Created ChosenAppleVectors object: {self}")

    def __str__(self) -> str:
        return f"ChosenAppleVectors(green_apple_vector={self.green_apple_vector}, "\
               f"winning_red_apple_vector={self.winning_red_apple_vector}, "\
               f"losing_red_apple_vectors={self.losing_red_apple_vectors})"

@dataclass
class ChosenAppleVectorsExtra:
    green_apple_vector: np.ndarray
    winning_red_apple_vector: np.ndarray
    losing_red_apple_vectors: np.ndarray
    green_apple_vector_extra: np.ndarray
    winning_red_apple_vector_extra: np.ndarray
    losing_red_apple_vectors_extra: np.ndarray

    def __post_init__(self) -> None:
        logging.debug(f"Created ChosenAppleVectorsExtra object: {self}")

    def __str__(self) -> str:
        return f"ChosenAppleVectorsExtra(green_apple_vector={self.green_apple_vector}, "\
               f"winning_red_apple_vector={self.winning_red_apple_vector}, "\
               f"losing_red_apple_vectors={self.losing_red_apple_vectors}, "\
               f"green_apple_vector_extra={self.green_apple_vector_extra}, "\
               f"winning_red_apple_vector_extra={self.winning_red_apple_vector_extra}, "\
               f"losing_red_apple_vectors_extra={self.losing_red_apple_vectors_extra})"


@dataclass
class GameState:
    number_of_players: int
    players: list["Agent"]
    max_cards_in_hand: int
    points_to_win: int
    total_games: int
    current_game: int
    current_round: int
    apples_in_play: ApplesInPlay | None
    chosen_apples: ChosenApples | None
    discard_pile: list[ChosenApples]
    current_judge: "Agent | None"
    round_winner: "Agent | None"
    game_winner: "Agent | None"

    def __post_init__(self) -> None:
        logging.debug(f"Created GameState object: {self}")

    def __str__(self) -> str:
        return f"GameState("\
                f"number_of_players={self.number_of_players}, "\
                f"players={[player.get_name() for player in self.players]}, "\
                f"max_cards_in_hand={self.max_cards_in_hand}, "\
                f"points_to_win={self.points_to_win}, "\
                f"total_games={self.total_games}, "\
                f"current_game={self.current_game}, "\
                f"current_round={self.current_round}, "\
                f"apples_in_play={self.apples_in_play}, "\
                f"chosen_apples={self.chosen_apples}, "\
                f"current_judge={self.current_judge.get_name() if self.current_judge is not None else None}, "\
                f"round_winner={self.round_winner.get_name() if self.round_winner is not None else None}, "\
                f"game_winner={self.game_winner.get_name() if self.game_winner is not None else None})"

    def to_dict(self) -> dict[str, list[str] | str | int | ApplesInPlay | ChosenApples | None]:
        return {
            "number_of_players": self.number_of_players,
            "players": [player.get_name() for player in self.players],
            "max_cards_in_hand": self.max_cards_in_hand,
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "apples_in_play": self.apples_in_play if self.apples_in_play is not None else None,
            "chosen_apples": self.chosen_apples if self.chosen_apples is not None else None,
            "current_judge": self.current_judge.get_name() if self.current_judge is not None else None,
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None,
            "game_winner": self.game_winner.get_name() if self.game_winner is not None else None
        }

    def gameplay_to_dict(self) -> dict[str, list[str] | str | int | ApplesInPlay | ChosenApples | None]:
        return {
            "number_of_players": self.number_of_players,
            "players": [player.get_name() for player in self.players],
            "max_cards_in_hand": self.max_cards_in_hand,
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "apples_in_play": self.apples_in_play if self.apples_in_play is not None else None,
            "chosen_apples": self.chosen_apples if self.chosen_apples is not None else None,
            "current_judge": self.current_judge.get_name() if self.current_judge is not None else None,
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None
        }

    def winner_to_dict(self) -> dict[str, str | None]:
        return {
            "Game Winner": self.game_winner.get_name() if self.game_winner is not None else None
        }

    def training_to_dict(self) -> dict[str, list[str] | str | int | ApplesInPlay | ChosenApples | None]:
        return {
            "number_of_players": self.number_of_players,
            "players": [player.get_name() for player in self.players],
            "max_cards_in_hand": self.max_cards_in_hand,
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "apples_in_play": self.apples_in_play if self.apples_in_play is not None else None,
            "chosen_apples": self.chosen_apples if self.chosen_apples is not None else None,
            "current_judge": self.current_judge.get_name() if self.current_judge is not None else None,
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None,
            "game_winner": self.game_winner.get_name() if self.game_winner is not None else None
        }


@dataclass
class PreferenceUpdates:
    agent: "Agent"
    round: int
    datetime: str
    green_apple: "GreenApple"
    winning_red_apple: "RedApple"
    slope: np.ndarray
    bias: np.ndarray

    def __str__(self) -> str:
        return f"PreferenceUpdates(agent={self.agent.get_name()}, round={self.round}, datetime={self.datetime}, "\
               f"green apple={self.green_apple.get_adjective()}, winning red apple={self.winning_red_apple.get_noun()}, "\
               f"slope={self.slope}), bias={self.bias}"

    def to_dict(self) -> dict:
        return {
            "Agent": self.agent.get_name(),
            "round": self.round,
            "datetime": self.datetime,
            "green_apple": self.green_apple.get_adjective(),
            "winning_red_apple": self.winning_red_apple.get_noun(),
            "slope": f"{self.slope}\n",
            "bias": f"{self.bias}\n"
        }


if __name__ == "__main__":
    pass
