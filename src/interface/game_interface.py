# Description: Abstract interface for Apples to Apples user interactions.

# Standard Libraries
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.interface.input.input_handler import InputHandler
    from src.interface.output.output_handler import OutputHandler


class GameInterface(ABC):
    """Abstract interface providing handlers for user interactions."""
    def __init__(self, input_handler: "InputHandler", output_handler: "OutputHandler") -> None:
        """Initialize the GameInterface."""
        self._input_handler: "InputHandler" = input_handler
        self._output_handler: "OutputHandler" = output_handler

    @property
    def input_handler(self) -> "InputHandler":
        """Return the input handler for this interface."""
        return self._input_handler

    @property
    def output_handler(self) -> "OutputHandler":
        """Return the output handler for this interface."""
        return self._output_handler

    @input_handler.setter
    def input_handler(self, input_handler: "InputHandler") -> None:
        """Set the input handler for this interface."""
        self._input_handler = input_handler

    @output_handler.setter
    def output_handler(self, output_handler: "OutputHandler") -> None:
        """Set the output handler for this interface."""
        self._output_handler = output_handler

    @abstractmethod
    def display_new_game_message(self) -> None:
        """Display a message indicating a new game is starting."""
        pass

    @abstractmethod
    def display_game_header(self, game_number: int, total_games: int) -> None:
        """Display the game header with game number information."""
        pass

    @abstractmethod
    def display_initializing_decks(self) -> None:
        """Display message about initializing decks."""
        pass

    @abstractmethod
    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        """Display the sizes of the green and red apple decks."""
        pass

    @abstractmethod
    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about a deck being loaded."""
        pass

    @abstractmethod
    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about an expansion deck being loaded."""
        pass

    @abstractmethod
    def display_initializing_players(self) -> None:
        """Display message about initializing players."""
        pass

    @abstractmethod
    def display_player_count(self, count: int) -> None:
        """Display the number of players in the game."""
        pass

    @abstractmethod
    def display_starting_judge(self, judge_name: str) -> None:
        """Display the name of the starting judge."""
        pass

    @abstractmethod
    def display_next_judge(self, judge_name: str) -> None:
        """Display the name of the next judge."""
        pass

    @abstractmethod
    def display_round_header(self, round_number: int) -> None:
        """Display the round header with round number information."""
        pass

    @abstractmethod
    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        """Display all players' points."""
        pass

    @abstractmethod
    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        """Display the green apple card in play."""
        pass

    @abstractmethod
    def display_player_red_apples(self, player: "Agent") -> None:
        """Display a player's red apples."""
        pass

    @abstractmethod
    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """Display that a player has chosen a red apple."""
        pass

    @abstractmethod
    def display_winning_red_apple(self, judge: "Agent", red_apple: "RedApple") -> None:
        """Display the winning red apple card."""
        pass

    @abstractmethod
    def display_round_winner(self, winner: "Agent") -> None:
        """Display the round winner."""
        pass

    @abstractmethod
    def display_game_winner(self, winner: "Agent") -> None:
        """Display the game winner."""
        pass

    @abstractmethod
    def display_game_time(self, minutes: int, seconds: int) -> None:
        """Display the total game time."""
        pass

    @abstractmethod
    def display_resetting_models(self) -> None:
        """Display message about resetting AI opponent models."""
        pass

    @abstractmethod
    def display_training_green_apple(self, adjective: str) -> None:
        """Display the green apple card in training mode."""
        pass

    # === Prompt Methods ===

    @abstractmethod
    def prompt_keep_players_between_games(self) -> bool:
        """Prompt whether to keep the same players between games."""
        pass

    @abstractmethod
    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the starting judge (returns 1-based index)."""
        pass

    @abstractmethod
    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player (1:Human, 2:Random, 3:AI)."""
        pass

    @abstractmethod
    def prompt_human_player_name(self) -> str:
        """Prompt for the human player's name."""
        pass

    @abstractmethod
    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type (1:Linear Regression, 2:Neural Network)."""
        pass

    @abstractmethod
    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode."""
        pass

    @abstractmethod
    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype (1:Literalist, 2:Contrarian, 3:Comedian)."""
        pass

    @abstractmethod
    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode."""
        pass

    @abstractmethod
    def prompt_judge_draw_green_apple(self, judge: "Agent") -> None:
        """Prompt the judge to draw a green apple."""
        pass

    @abstractmethod
    def prompt_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a red apple, returning the index."""
        pass

    @abstractmethod
    def prompt_training_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a good red apple in training mode."""
        pass

    @abstractmethod
    def prompt_training_select_bad_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a bad red apple in training mode."""
        pass

    @abstractmethod
    def prompt_judge_select_winner(self, judge: "Agent") -> None:
        """Prompt the judge to select the winning red apple."""
        pass
