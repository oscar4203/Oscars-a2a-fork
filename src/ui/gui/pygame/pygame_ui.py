# Description: Pygame-based UI implementation for Apples to Apples.

# Standard Libraries
import pygame
import logging
import time
from typing import List, TYPE_CHECKING, Tuple

# Local Modules
from src.interface.game_interface import GameInterface

# Disable Pygame Audio Driver (to prevent ALSA errors)
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.interface.input.input_handler import InputHandler
    from src.interface.output.output_handler import OutputHandler


# Constants
BACKGROUND_COLOR = (35, 106, 135)
TEXT_COLOR = (230, 230, 230)
GREEN_APPLE_COLOR = (100, 200, 100)
RED_APPLE_COLOR = (200, 100, 100)
JUDGE_COLOR = (200, 200, 100)
FONT_SIZE_LARGE = 36
FONT_SIZE_NORMAL = 24
FONT_SIZE_SMALL = 18

# Global gameplay delay time (seconds)
DELAY_TIME = 3.0


class PygameUI(GameInterface):
    """Pygame-based UI implementation for Apples to Apples."""

    def __init__(self, resolution: List[int] = [1200, 800], fps: int = 30):
        """Initialize the PygameUI with specified resolution and frame rate."""
        # Initialize pygame
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((resolution[0], resolution[1]))
            pygame.display.set_caption("Apples to Apples")

            # Try to set an icon if available
            try:
                icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
                if os.path.exists(icon_path):
                    programIcon = pygame.image.load(icon_path)
                    pygame.display.set_icon(programIcon)
            except Exception as e:
                logging.warning(f"Could not load GUI icon: {e}")

            # Initialize fonts
            self.font_normal = pygame.font.Font(None, FONT_SIZE_NORMAL)
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
            self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)

            # Create clock for FPS control
            self.clock = pygame.time.Clock()
            self.fps = fps

            # Game state flags
            self.running = False
            self.paused = False

            # Screen dimensions
            self.width = resolution[0]
            self.height = resolution[1]

            # Messages and notifications
            self.messages = []
            self.notification = None
            self.notification_time = 0

            # Create input and output handlers
            from src.interface.input.pygame_input_handler import PygameInputHandler
            from src.interface.output.pygame_output_handler import PygameOutputHandler

            self._input_handler = PygameInputHandler(self)
            self._output_handler = PygameOutputHandler(self)

            logging.info("Pygame UI initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Pygame UI: {e}", exc_info=True)
            raise

    @property
    def input_handler(self) -> "InputHandler":
        """Return the input handler for this interface."""
        return self._input_handler

    @property
    def output_handler(self) -> "OutputHandler":
        """Return the output handler for this interface."""
        return self._output_handler

    # Basic UI utility methods

    def add_message(self, message: str) -> None:
        """Add a message to the message queue (for display in the message area)."""
        self.messages.append(message)
        if len(self.messages) > 5:  # Keep only the most recent messages
            self.messages.pop(0)

    def show_notification(self, message: str, duration: float = DELAY_TIME) -> None:
        """Show a temporary notification on screen."""
        # During player setup, don't use auto-timing
        if "Select Player" in message or "Select type for Player" in message or "Enter Human Player Name" in message:
            self.notification = message
            self.notification_time = float("inf")
        else:
            self.notification = message
            self.notification_time = time.time() + duration

    # Drawing utilities for use by the output handler

    def draw_text(self, text: str, font, color, x, y, center=False):
        """Helper function to draw text on the screen."""
        try:
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect()
            if center:
                text_rect.center = (x, y)
            else:
                text_rect.topleft = (x, y)
            self.screen.blit(text_surface, text_rect)
        except Exception as e:
            logging.error(f"Error rendering text '{text}': {e}")

    def draw_rect(self, color, rect, border_radius=0, width=0):
        """Helper to draw a rectangle."""
        pygame.draw.rect(self.screen, color, rect, width, border_radius=border_radius)

    def draw_notification(self):
        """Draw current notification if active."""
        if self.notification and time.time() < self.notification_time:
            # Draw semi-transparent background
            notification_surface = pygame.Surface((self.width - 100, 60), pygame.SRCALPHA)
            notification_surface.fill((0, 0, 0, 180))  # Black with alpha
            self.screen.blit(notification_surface, (50, self.height // 2 - 30))

            # Draw notification text
            self.draw_text(
                self.notification,
                self.font_normal,
                (255, 255, 255),
                self.width // 2,
                self.height // 2,
                center=True
            )
        elif self.notification and time.time() >= self.notification_time:
            self.notification = None

    # Main UI lifecycle methods

    def update_display(self):
        """Update the pygame display."""
        pygame.display.flip()
        self.clock.tick(self.fps)

    def process_events(self):
        """Process pygame events. Returns False if the game should exit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            # Process input events
            from src.interface.input.pygame_input_handler import PygameInputHandler
            if isinstance(self._input_handler, PygameInputHandler):
                self._input_handler.process_event(event)

        return True

    def delay_for_visibility(self, seconds: float = DELAY_TIME):
        """Pause execution while keeping the UI responsive."""
        start_time = time.time()
        while time.time() - start_time < seconds:
            if not self.process_events():
                break

            # Clear screen and redraw during the delay
            self.screen.fill(BACKGROUND_COLOR)

            # Draw game state
            from src.interface.output.pygame_output_handler import PygameOutputHandler
            if isinstance(self._output_handler, PygameOutputHandler):
                self._output_handler.draw_game_state()

            self.draw_notification()

            self.update_display()
            pygame.time.wait(50)  # Short wait to reduce CPU usage

    def run(self) -> None:
        """Start the main pygame event loop."""
        self.running = True
        logging.info("Starting Pygame main loop")

        while self.running:
            # Process events
            if not self.process_events():
                break

            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)

            # Draw game state
            from src.interface.output.pygame_output_handler import PygameOutputHandler
            if isinstance(self._output_handler, PygameOutputHandler):
                self._output_handler.draw_game_state()

            # Draw notifications on top of everything
            self.draw_notification()

            # Update display
            self.update_display()

        # Clean up pygame
        pygame.quit()
        logging.info("Pygame main loop ended")

    def teardown(self) -> None:
        """Clean up pygame resources."""
        self.running = False
        pygame.quit()
        logging.info("Pygame resources cleaned up")

# GameInterface method implementations - these delegate to the handlers

    def display_new_game_message(self) -> None:
        """Display a message indicating a new game is starting."""
        self._output_handler.display_new_game_message()

    def display_game_header(self, game_number: int, total_games: int) -> None:
        """Display the game header with game number information."""
        self._output_handler.display_game_header(game_number, total_games)

    def display_initializing_decks(self) -> None:
        """Display message about initializing decks."""
        self._output_handler.display_initializing_decks()

    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        """Display the sizes of the green and red apple decks."""
        self._output_handler.display_deck_sizes(green_deck_size, red_deck_size)

    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about a deck being loaded."""
        self._output_handler.display_deck_loaded(deck_name, count)

    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about an expansion deck being loaded."""
        self._output_handler.display_expansion_deck_loaded(deck_name, count)

    def display_initializing_players(self) -> None:
        """Display message about initializing players."""
        self._output_handler.display_initializing_players()

    def display_player_count(self, count: int) -> None:
        """Display the number of players in the game."""
        self._output_handler.display_player_count(count)

    def display_starting_judge(self, judge_name: str) -> None:
        """Display the name of the starting judge."""
        self._output_handler.display_starting_judge(judge_name)

    def display_next_judge(self, judge_name: str) -> None:
        """Display the name of the next judge."""
        self._output_handler.display_next_judge(judge_name)

    def display_round_header(self, round_number: int) -> None:
        """Display the round header with round number information."""
        self._output_handler.display_round_header(round_number)

    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        """Display all players' points."""
        self._output_handler.display_player_points(player_points)

    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        """Display the green apple card in play."""
        self._output_handler.display_green_apple(judge, green_apple)

    def display_player_red_apples(self, player: "Agent") -> None:
        """Display a player's red apples."""
        self._output_handler.display_player_red_apples(player)

    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """Display that a player has chosen a red apple."""
        self._output_handler.display_red_apple_chosen(player, red_apple)

    def display_winning_red_apple(self, judge: "Agent", red_apple: "RedApple") -> None:
        """Display the winning red apple card."""
        self._output_handler.display_winning_red_apple(judge, red_apple)

    def display_round_winner(self, winner: "Agent") -> None:
        """Display the round winner."""
        self._output_handler.display_round_winner(winner)

    def display_game_winner(self, winner: "Agent") -> None:
        """Display the game winner."""
        self._output_handler.display_game_winner(winner)

    def display_game_time(self, minutes: int, seconds: int) -> None:
        """Display the total game time."""
        self._output_handler.display_game_time(minutes, seconds)

    def display_resetting_models(self) -> None:
        """Display message about resetting AI opponent models."""
        self._output_handler.display_resetting_models()

    def display_training_green_apple(self, green_apple: "GreenApple") -> None:
        """Display the green apple card in training mode."""
        self._output_handler.display_training_green_apple(green_apple)

    # === Prompt Methods ===

    def prompt_keep_players_between_games(self) -> bool:
        """Prompt whether to keep the same players between games."""
        return self._input_handler.prompt_yes_no("Keep the same players for the next game?")

    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the starting judge."""
        return self._input_handler.prompt_starting_judge(player_count)

    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player."""
        return self._input_handler.prompt_player_type(player_number)

    def prompt_human_player_name(self) -> str:
        """Prompt for the human player's name."""
        return self._input_handler.prompt_human_player_name()

    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type."""
        return self._input_handler.prompt_ai_model_type()

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode."""
        return self._input_handler.prompt_training_model_type()

    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype."""
        return self._input_handler.prompt_ai_archetype()

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode."""
        return self._input_handler.prompt_training_pretrained_type()

    def prompt_judge_draw_green_apple(self, judge: "Agent") -> None:
        """Prompt the judge to draw a green apple."""
        self._output_handler.prompt_judge_draw_green_apple(judge)

    def prompt_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a red apple."""
        self._output_handler.prompt_select_red_apple(player, green_apple)

    def prompt_training_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a good red apple in training mode."""
        self._output_handler.prompt_training_select_red_apple(player, green_apple)

    def prompt_training_select_bad_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Prompt a player to select a bad red apple in training mode."""
        self._output_handler.prompt_training_select_bad_red_apple(player, green_apple)

    def prompt_judge_select_winner(self, judge: "Agent") -> None:
        """Prompt the judge to select the winning red apple."""
        self._output_handler.prompt_judge_select_winner(judge)
