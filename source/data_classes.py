# Description: Data classes for the game state and preference updates.

# Standard Libraries
import logging
import numpy as np
from dataclasses import dataclass

# Third-party Libraries

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent


@dataclass
class GameState:
    number_of_players: int
    players: list[Agent]
    max_cards_in_hand: int
    points_to_win: int
    total_games: int
    current_game: int
    current_round: int
    green_apple: dict[Agent, GreenApple] | None
    red_apples: list[dict[Agent, RedApple]]
    winning_red_apple: RedApple | None
    losing_red_apples: list[RedApple]
    current_judge: Agent | None
    round_winner: Agent | None
    game_winner: Agent | None

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
                f"green_apple={self.green_apple[self.current_judge].get_adjective() if self.green_apple is not None and self.current_judge is not None else None}, "\
                f"red_apples={[{player.get_name(): apple.get_noun()} for entry in self.red_apples for player, apple in entry.items()]}, "\
                f"winning_red_apple={self.winning_red_apple.get_noun() if self.winning_red_apple is not None else None}, "\
                f"losing_red_apples={[apple.get_noun() for apple in self.losing_red_apples]}, "\
                f"current_judge={self.current_judge.get_name() if self.current_judge is not None else None}, "\
                f"round_winner={self.round_winner.get_name() if self.round_winner is not None else None}, "\
                f"game_winner={self.game_winner.get_name() if self.game_winner is not None else None})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict[str, list[str] | str | int | list[dict[str, str]] | None]:
        return {
            "number_of_players": self.number_of_players,
            "players": [player.get_name() for player in self.players],
            "max_cards_in_hand": self.max_cards_in_hand,
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "green_apple": self.green_apple[self.current_judge].get_adjective() if self.green_apple is not None and self.current_judge is not None else None,
            "red_apples": [{player.get_name(): apple.get_noun()} for entry in self.red_apples for player, apple in entry.items()],
            "winning_red_apple": self.winning_red_apple.get_noun() if self.winning_red_apple is not None else None,
            "losing_red_apples": [apple.get_noun() for apple in self.losing_red_apples],
            "current_judge": self.current_judge.get_name() if self.current_judge is not None else None,
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None,
            "game_winner": self.game_winner.get_name() if self.game_winner is not None else None
        }

    def gameplay_to_dict(self) -> dict[str, list[str] | str | int | list[dict[str, str]] | None]:
        return {
            "number_of_players": self.number_of_players,
            "players": [player.get_name() for player in self.players],
            "max_cards_in_hand": self.max_cards_in_hand,
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "green_apple": self.green_apple[self.current_judge].get_adjective() if self.green_apple is not None and self.current_judge is not None else None,
            "red_apples": [{player.get_name(): apple.get_noun()} for entry in self.red_apples for player, apple in entry.items()],
            "winning_red_apple": self.winning_red_apple.get_noun() if self.winning_red_apple is not None else None,
            "losing_red_apples": [apple.get_noun() for apple in self.losing_red_apples],
            "current_judge": self.current_judge.get_name() if self.current_judge is not None else None,
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None
        }

    def winner_to_dict(self) -> dict[str, str | None]:
        return {
            "Game Winner": self.game_winner.get_name() if self.game_winner is not None else None
        }

    def training_to_dict(self) -> dict[str, list[str] | str | int | list[dict[str, str]] | None]:
        return {
            "number_of_players": self.number_of_players,
            "players": [player.get_name() for player in self.players],
            "max_cards_in_hand": self.max_cards_in_hand,
            "points_to_win": self.points_to_win,
            "total_games": self.total_games,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "green_apple": self.green_apple[self.current_judge].get_adjective() if self.green_apple is not None and self.current_judge is not None else None,
            "red_apples": [{player.get_name(): apple.get_noun()} for entry in self.red_apples for player, apple in entry.items()],
            "winning_red_apple": self.winning_red_apple.get_noun() if self.winning_red_apple is not None else None,
            "losing_red_apples": [apple.get_noun() for apple in self.losing_red_apples],
            "current_judge": self.current_judge.get_name() if self.current_judge is not None else None,
            "round_winner": self.round_winner.get_name() if self.round_winner is not None else None,
            "game_winner": self.game_winner.get_name() if self.game_winner is not None else None
        }


@dataclass
class ModelData:
    green_apples: list[GreenApple]
    red_apples: list[RedApple]
    winning_red_apples: list[RedApple]

    def __post_init__(self) -> None:
        logging.debug(f"Created ModelData object: {self}")

    def __str__(self) -> str:
        return f"ModelData(green_apples={[apple.__adjective for apple in self.green_apples]}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apples={[apple.get_noun() for apple in self.red_apples]})"

    def __repr__(self) -> str:
        return f"ModelData(green_apples={[apple.__adjective for apple in self.green_apples]}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apples={[apple.get_noun() for apple in self.red_apples]})"

    def to_dict(self) -> dict:
        return {
            "green_apples": [apple.__adjective for apple in self.green_apples],
            "red_apples": [apple.get_noun() for apple in self.red_apples],
            "winning_red_apples": [apple.get_noun() for apple in self.winning_red_apples]
        }


@dataclass
class PreferenceUpdates:
    agent: Agent
    round: int
    datetime: str
    green_apple: GreenApple
    winning_red_apple: RedApple
    slope: np.ndarray
    bias: np.ndarray

    def __str__(self) -> str:
        return f"PreferenceUpdates(agent={self.agent.get_name()}, round={self.round}, datetime={self.datetime}, "\
               f"green apple={self.green_apple.get_adjective()}, winning red apple={self.winning_red_apple.get_noun()}, "\
               f"slope={self.slope}), bias={self.bias}"

    def __repr__(self) -> str:
         return self.__str__()

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
