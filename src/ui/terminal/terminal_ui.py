# Description: Terminal-based UI implementation for Apples to Apples.

# Standard Libraries
from typing import List, Tuple, TYPE_CHECKING

# Local Modules
from src.interface.game_interface import GameInterface

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.interface.input.input_handler import InputHandler
    from src.interface.output.output_handler import OutputHandler


class TerminalUI(GameInterface):
    """Terminal-based user interface implementation for Apples to Apples."""

    def __init__(self, input_handler: "InputHandler", output_handler: "OutputHandler"):
        """
        Initialize the terminal UI.

        Args:
            print_in_terminal: Whether to print output to the terminal
        """
        self.input_handler = input_handler
        self.output_handler = output_handler

    # === Display Methods ===

    def display_new_game_message(self) -> None:
        """Display a message indicating a new game is starting."""
        self.output_handler.display_new_game_message()

    def display_game_header(self, game_number: int, total_games: int) -> None:
        """Display the game header with game number information."""
        self.output_handler.display_game_header(game_number, total_games)

    def display_initializing_decks(self) -> None:
        """Display message about initializing decks."""
        self.output_handler.display_initializing_decks()

    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        """Display the sizes of the green and red apple decks."""
        self.output_handler.display_deck_sizes(green_deck_size, red_deck_size)

    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about a deck being loaded."""
        self.output_handler.display_deck_loaded(deck_name, count)

    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about an expansion deck being loaded."""
        self.output_handler.display_expansion_deck_loaded(deck_name, count)

    def display_initializing_players(self) -> None:
        """Display message about initializing players."""
        self.output_handler.display_initializing_players()

    def display_player_count(self, count: int) -> None:
        """Display the number of players in the game."""
        self.output_handler.display_player_count(count)

    def display_starting_judge(self, judge_name: str) -> None:
        """Display the name of the starting judge."""
        self.output_handler.display_starting_judge(judge_name)

    def display_next_judge(self, judge_name: str) -> None:
        """Display the name of the next judge."""
        self.output_handler.display_next_judge(judge_name)

    def display_round_header(self, round_number: int) -> None:
        """Display the round header with round number information."""
        self.output_handler.display_round_header(round_number)

    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        """Display all players' points."""
        self.output_handler.display_player_points(player_points)

    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        """Display the green apple card in play."""
        self.output_handler.display_green_apple(judge, green_apple)

    def display_player_red_apples(self, player: "Agent") -> None:
        """Display the red apples held by a player."""
        self.output_handler.display_player_red_apples(player)

    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """Display the red apple chosen by a player."""
        self.output_handler.display_red_apple_chosen(player, red_apple)

    def display_winning_red_apple(self, judge: "Agent", red_apple: "RedApple") -> None:
        """Display the winning red apple card."""
        self.output_handler.display_winning_red_apple(judge, red_apple)

    def display_round_winner(self, winner: "Agent") -> None:
        """Display the round winner."""
        self.output_handler.display_round_winner(winner)

    def display_game_winner(self, winner: "Agent") -> None:
        """Display the game winner."""
        self.output_handler.display_game_winner(winner)

    def display_game_time(self, minutes: int, seconds: int) -> None:
        """Display the total game time."""
        self.output_handler.display_game_time(minutes, seconds)

    def display_resetting_models(self) -> None:
        """Display message about resetting AI opponent models."""
        self.output_handler.display_resetting_models()

    def display_training_green_apple(self, green_apple: "GreenApple") -> None:
        """Display the green apple card in training mode."""
        self.output_handler.display_training_green_apple(green_apple)

    # === Prompt Methods ===

    def prompt_keep_players_between_games(self) -> bool:
        """Prompt whether to keep the same players between games."""
        return self.input_handler.prompt_yes_no("Do you want to keep the same players as last game?")

    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the starting judge (returns 1-based index)."""
        return self.input_handler.prompt_starting_judge(player_count)

    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player (1:Human, 2:Random, 3:AI)."""
        return self.input_handler.prompt_player_type(player_number)

    def prompt_human_player_name(self) -> str:
        """Prompt for the human player's name."""
        return self.input_handler.prompt_human_player_name()

    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type (1:Linear Regression, 2:Neural Network)."""
        return self.input_handler.prompt_ai_model_type()

    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype (1:Literalist, 2:Contrarian, 3:Comedian)."""
        return self.input_handler.prompt_ai_archetype()

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode."""
        return self.input_handler.prompt_training_model_type()

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode."""
        return self.input_handler.prompt_training_pretrained_type()

    def prompt_judge_draw_green_apple(self, judge: "Agent") -> None:
        """Prompt the judge to draw a green apple."""
        self.output_handler.prompt_judge_draw_green_apple(judge)

    def prompt_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a red apple."""
        self.output_handler.prompt_select_red_apple(player, green_apple)

    def prompt_training_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a good red apple in training mode."""
        self.output_handler.prompt_training_select_red_apple(player, green_apple)

    def prompt_training_select_bad_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a bad red apple in training mode."""
        self.output_handler.prompt_training_select_bad_red_apple(player, green_apple)

    def prompt_judge_select_winner(self, judge: "Agent") -> None:
        """Prompt the judge to select the winning red apple."""
        self.output_handler.prompt_judge_select_winner(judge)
