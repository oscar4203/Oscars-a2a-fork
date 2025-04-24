import pygame
import logging
import sys
from typing import TYPE_CHECKING

# Add project root to sys.path if necessary
# (May not be needed if game_driver runs from root, but good practice)
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable Pygame Audio Driver (to prevent ALSA errors)
# This needs to be done BEFORE pygame.init()
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Type Hinting Imports
if TYPE_CHECKING:
    from src.apples_to_apples import ApplesToApples
    from src.data_classes.data_classes import GameLog, GreenApple, RedApple
    from src.agent_model.agent import Agent

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (230, 230, 230)
GREEN_APPLE_COLOR = (100, 200, 100)
RED_APPLE_COLOR = (200, 100, 100)
JUDGE_COLOR = (200, 200, 100)
FONT_SIZE_NORMAL = 24
FONT_SIZE_LARGE = 36
FONT_SIZE_SMALL = 18

class PygameGUIWrapper:
    def __init__(self, game: "ApplesToApples"):
        """Initializes the Pygame GUI wrapper."""
        self.game: "ApplesToApples" = game
        self.screen: pygame.Surface | None = None
        self.font_normal: pygame.font.Font | None = None
        self.font_large: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None
        self.clock: pygame.time.Clock | None = None
        self.running = False

        # Basic Pygame Initialization
        try:
            # Pygame init should now avoid ALSA errors due to the environment variable
            pygame.init()
            # Attempt to set icon (replace 'icon.png' with your actual icon file if you have one)
            try:
                icon_path = os.path.join(project_root, "assets", "icon.png") # Example path
                if os.path.exists(icon_path):
                    programIcon = pygame.image.load(icon_path)
                    pygame.display.set_icon(programIcon)
                else:
                    logging.warning("GUI Icon not found at expected path.")
            except Exception as e:
                logging.warning(f"Could not load GUI icon: {e}")

            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Apples to Apples AI Agent")
            self.font_normal = pygame.font.Font(None, FONT_SIZE_NORMAL) # Default font
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
            self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
            self.clock = pygame.time.Clock()
            logging.info("Pygame initialized successfully.")
        except Exception as e:
            logging.error(f"Pygame initialization failed: {e}", exc_info=True)
            raise # Re-raise the exception to stop execution

    def _draw_text(self, text: str, font: pygame.font.Font | None, color: tuple, x: int, y: int, center: bool = False):
        """Helper function to draw text."""
        # Check for screen and also if the passed font is None
        if not self.screen or not font:
            return
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

    def _draw_game_info(self):
        """Draws general game info (Game #, Round #, Judge)."""
        if not self.screen or not self.font_normal or not self.font_large: return
        try:
            game_log: "GameLog" = self.game.get_game_log()
            game_num = game_log.get_current_game_number()
            total_games = game_log.total_games
            round_num = game_log.get_current_round_number()
            judge: "Agent | None" = game_log.get_current_judge()

            self._draw_text(f"Game: {game_num}/{total_games}", self.font_normal, TEXT_COLOR, 10, 10)
            self._draw_text(f"Round: {round_num}", self.font_normal, TEXT_COLOR, 10, 40)
            judge_text = f"Judge: {judge.get_name()}" if judge else "Judge: (None)"
            self._draw_text(judge_text, self.font_large, JUDGE_COLOR, SCREEN_WIDTH // 2, 30, center=True)

        except Exception as e:
            logging.warning(f"Could not draw game info: {e}")
            self._draw_text("Error loading game info", self.font_normal, (255,0,0), 10, 10)

    def _draw_green_apple(self):
        """Draws the current green apple."""
        if not self.screen or not self.font_large or not self.font_small: return
        y_pos = 80
        try:
            game_log: "GameLog" = self.game.get_game_log()
            apples_in_play = game_log.get_apples_in_play()
            green_apple: "GreenApple | None" = apples_in_play.get_green_apple() if apples_in_play else None

            if green_apple:
                self._draw_text(green_apple.get_adjective(), self.font_large, GREEN_APPLE_COLOR, SCREEN_WIDTH // 2, y_pos, center=True)
                desc = green_apple.get_synonyms() if hasattr(green_apple, 'get_synonyms') else green_apple.get_synonyms()
                self._draw_text(f"({desc})", self.font_small, GREEN_APPLE_COLOR, SCREEN_WIDTH // 2, y_pos + 30, center=True)
            else:
                self._draw_text("(No Green Apple Drawn)", self.font_normal, TEXT_COLOR, SCREEN_WIDTH // 2, y_pos, center=True)

        except Exception as e:
            logging.warning(f"Could not draw green apple: {e}")
            self._draw_text("Error loading green apple", self.font_normal, (255,0,0), SCREEN_WIDTH // 2, y_pos, center=True)

    def _draw_players(self):
        """Draws player names and scores."""
        if not self.screen or not self.font_normal: return
        start_y = 150
        x_offset = 20
        y_gap = 30
        try:
            game_log: "GameLog" = self.game.get_game_log()
            players = game_log.get_game_players()
            if not players:
                self._draw_text("No players", self.font_normal, TEXT_COLOR, x_offset, start_y)
                return

            for i, player in enumerate(players):
                color = JUDGE_COLOR if player.get_judge_status() else TEXT_COLOR
                self._draw_text(f"{player.get_name()}: {player.get_points()}", self.font_normal, color, x_offset, start_y + i * y_gap)

        except Exception as e:
            logging.warning(f"Could not draw players: {e}")
            self._draw_text("Error loading players", self.font_normal, (255,0,0), x_offset, start_y)

    # Placeholder for drawing submitted cards
    def _draw_submitted_cards(self):
        if not self.screen or not self.font_normal: return
        start_y = 150
        start_x = 300 # Example position
        y_gap = 60
        card_width = 200
        card_height = 50
        self._draw_text("Submitted Cards:", self.font_normal, TEXT_COLOR, start_x, start_y - 30)
        try:
            game_log: "GameLog" = self.game.get_game_log()
            apples_in_play = game_log.get_apples_in_play()
            submitted_dicts = apples_in_play.red_apples if apples_in_play else []

            if not submitted_dicts:
                 self._draw_text("(None)", self.font_small, TEXT_COLOR, start_x, start_y)
                 return

            for i, card_dict in enumerate(submitted_dicts):
                player = next(iter(card_dict.keys()), None)
                card: "RedApple | None" = next(iter(card_dict.values()), None)
                if player and card:
                    # Simple text representation for now
                    card_text = f"{card.get_noun()}"
                    player_text = f"({player.get_name()})"
                    y_pos = start_y + i * y_gap
                    # Draw a simple rect as background
                    pygame.draw.rect(self.screen, RED_APPLE_COLOR, (start_x, y_pos, card_width, card_height), border_radius=5)
                    self._draw_text(card_text, self.font_normal, (0,0,0), start_x + card_width // 2, y_pos + card_height // 3, center=True)
                    self._draw_text(player_text, self.font_small, (50,50,50), start_x + card_width // 2, y_pos + card_height * 2 // 3, center=True)

        except Exception as e:
            logging.warning(f"Could not draw submitted cards: {e}")
            self._draw_text("Error loading submitted cards", self.font_normal, (255,0,0), start_x, start_y)


    # Placeholder for drawing player hands
    def _draw_player_hands(self):
        # This would involve drawing cards for each player, likely only showing the human player's hand clearly.
        # For now, we'll skip the detailed implementation.
        pass

    def run(self):
        """Runs the main Pygame event loop."""
        if not self.screen or not self.clock:
            logging.error("Pygame screen or clock not initialized. Cannot run GUI.")
            return

        self.running = True
        while self.running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # Add more event handling here (keyboard, mouse clicks) later

            # Game Logic Update (Placeholder)
            # If the game needs to advance based on time or state, do it here.
            # For turn-based, updates will likely be triggered by events (e.g., button clicks).
            # Example: Check if game needs to start
            try:
                if self.game.get_game_log().get_current_round_number() == 0:
                    # Maybe show a "Start Game" button or automatically start?
                    # For now, let's assume game_driver called new_game() if needed,
                    # or we need a button here. Let's just display.
                    pass
            except Exception:
                 pass # Ignore errors if game log not ready

            # Drawing
            self.screen.fill(BACKGROUND_COLOR) # Clear screen

            # Draw various elements
            self._draw_game_info()
            self._draw_green_apple()
            self._draw_players()
            self._draw_submitted_cards()
            self._draw_player_hands() # Call placeholder

            # Update Display
            pygame.display.flip()

            # Frame Rate Control
            self.clock.tick(30) # Limit FPS

        # Pygame Quit
        logging.info("Exiting Pygame.")
        pygame.quit()

# Main Execution Block (for testing wrapper directly if needed)
if __name__ == "__main__":
    print("This script is intended to be launched by game_driver.py using the -G flag.")
    print("For testing, you might need to create a mock ApplesToApples object.")
    # Example (requires creating mock objects or a simplified setup):
    # class MockEmbedding: pass
    # class MockGameLog:
    #     def get_current_game_number(self): return 1
    #     def total_games(self): return 1
    #     def get_current_round_number(self): return 0
    #     def get_current_judge(self): return None
    #     def get_apples_in_play(self): return None
    #     def get_game_players(self): return []
    # class MockApplesToApples:
    #     def __init__(self): self._game_log = MockGameLog()
    #     def get_game_log(self): return self._game_log
    #
    # try:
    #     mock_game = MockApplesToApples()
    #     wrapper = PygameGUIWrapper(game=mock_game)
    #     wrapper.run()
    # except Exception as e:
    #     print(f"Error during test run: {e}")
