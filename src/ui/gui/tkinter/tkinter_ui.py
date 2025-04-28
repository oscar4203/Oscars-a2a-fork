# Description: Tkinter-based UI implementation for Apples to Apples.

# Standard Libraries
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Dict, Optional, Callable, TYPE_CHECKING

# Local Modules
from src.interface.game_interface import GameInterface
from src.interface.input.tkinter_input_handler import TkinterInputHandler
from src.interface.output.tkinter_output_handler import TkinterOutputHandler
from src.ui.gui.tkinter.tkinter_widgets import PlayerInfoWidget, RedAppleCard

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.core.state import GameStateManager


class TkinterUI(GameInterface):
    """Tkinter-based user interface implementation for Apples to Apples."""

    # @property
    # def input_handler(self) -> "InputHandler":
    #     return self.input_handler

    # @property
    # def output_handler(self) -> "OutputHandler":
    #     return self.output_handler

    def __init__(self, #input_handler: "InputHandler", output_handler: "OutputHandler",
                 state_manager: Optional["GameStateManager"] = None):
        """Initialize the Tkinter UI."""
        self.root = tk.Tk()
        self.root.title("Apples to Apples")
        self.root.geometry("1024x768")

        # Reference to the state manager
        self.state_manager = state_manager

        # Card selection tracking - fix type annotations
        self.selected_card_index: Optional[int] = None
        self.active_cards: List[RedAppleCard] = []
        self.confirm_button: Optional[tk.Button] = None
        self.on_card_confirm: Optional[Callable[[], None]] = None

        # Set up the main UI structure
        self.setup_ui()

        # Create input and output handlers
        self.input_handler = TkinterInputHandler(self.root, self)
        self.output_handler = TkinterOutputHandler(self.root, self)

        # Player widget management
        self.player_widgets: Dict["Agent", PlayerInfoWidget] = {}

    def setup_ui(self):
        """Set up the main UI structure."""
        # Main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Header frame
        self.header_frame = ttk.Frame(self.main_frame, padding="5")
        self.header_frame.pack(fill=tk.X, pady=(0, 10))

        # Game title
        self.title_label = ttk.Label(self.header_frame, text="Apples to Apples", font=("Arial", 18, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=3)

        # Game info (game number, round number, points to win)
        self.game_info_frame = ttk.Frame(self.header_frame)
        self.game_info_frame.grid(row=1, column=0, columnspan=3, pady=5)

        self.game_label = ttk.Label(self.game_info_frame, text="Game: 0/0")
        self.game_label.grid(row=0, column=0, padx=10)

        self.round_label = ttk.Label(self.game_info_frame, text="Round: 0")
        self.round_label.grid(row=0, column=1, padx=10)

        self.points_label = ttk.Label(self.game_info_frame, text="Points to Win: 0")
        self.points_label.grid(row=0, column=2, padx=10)

        # Status message
        self.status_frame = ttk.Frame(self.main_frame, padding="5")
        self.status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(self.status_frame, text="Waiting to start game...", wraplength=800)
        self.status_label.pack()

        # Game content area
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Green apple area (top)
        self.green_apple_frame = ttk.Frame(self.content_frame, padding="5")
        self.green_apple_frame.pack(fill=tk.X, pady=(0, 10))

        self.green_apple_label = ttk.Label(self.green_apple_frame, text="Green Apple")
        self.green_apple_label.pack()

        self.green_apple_card_frame = ttk.Frame(self.green_apple_frame)
        self.green_apple_card_frame.pack(pady=5)

        # Players area (left, right, top)
        self.players_frame = ttk.Frame(self.content_frame)
        self.players_frame.pack(fill=tk.BOTH, expand=True)

        self.left_players_frame = ttk.Frame(self.players_frame, padding="5")
        self.left_players_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.center_frame = ttk.Frame(self.players_frame)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_players_frame = ttk.Frame(self.players_frame, padding="5")
        self.right_players_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Submitted red apples area (center)
        self.submitted_cards_frame = ttk.Frame(self.center_frame, padding="5")
        self.submitted_cards_frame.pack(fill=tk.BOTH, expand=True)

        self.submitted_cards_label = ttk.Label(self.submitted_cards_frame, text="Submitted Red Apples")
        self.submitted_cards_label.pack()

        self.submitted_cards_container = ttk.Frame(self.submitted_cards_frame)
        self.submitted_cards_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # Player's red apples area (bottom)
        self.player_cards_frame = ttk.Frame(self.main_frame, padding="5")
        self.player_cards_frame.pack(fill=tk.X, pady=(10, 0))

        self.player_cards_label = ttk.Label(self.player_cards_frame, text="Your Red Apples")
        self.player_cards_label.pack()

        self.player_cards_container = ttk.Frame(self.player_cards_frame)
        self.player_cards_container.pack(fill=tk.X, pady=5)

    def update_players(self, players):
        """Update the display of players."""
        self.players = players
        self.player_widgets = {}

        # Clear existing player widgets
        for frame in [self.left_players_frame, self.right_players_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        # Distribute players to left and right sides
        left_players = players[:len(players)//2]
        right_players = players[len(players)//2:]

        # Create player widgets
        for player in left_players:
            widget = PlayerInfoWidget(
                self.left_players_frame,
                player.get_name(),
                player.get_points(),
                player.get_judge_status()
            )
            widget.pack(pady=5, fill=tk.X)
            self.player_widgets[player] = widget

        for player in right_players:
            widget = PlayerInfoWidget(
                self.right_players_frame,
                player.get_name(),
                player.get_points(),
                player.get_judge_status()
            )
            widget.pack(pady=5, fill=tk.X)
            self.player_widgets[player] = widget

        self.root.update_idletasks()

    def run(self):
        """Run the Tkinter main loop."""
        self.root.mainloop()

    # === GameInterface implementation methods ===
    # These methods just delegate to the appropriate handler

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
        """Display a player's red apples."""
        self.output_handler.display_player_red_apples(player)

    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """Display that a player has chosen a red apple."""
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
