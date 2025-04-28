# Description: Pygame implementation of the input handler.

# Standard Libraries
import pygame
from typing import List, Dict, TYPE_CHECKING
import logging

# Local Modules
from src.interface.input.input_handler import InputHandler
from src.ui.gui.pygame.pygame_ui import BACKGROUND_COLOR

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.ui.gui.pygame.pygame_ui import PygameUI


class PygameInputHandler(InputHandler):
    """Implementation of the input handler for Pygame."""

    def __init__(self, ui: "PygameUI"):
        """Initialize the Pygame input handler."""
        self.ui = ui
        self.selected_index = None
        self.dialog_result = None
        self.waiting_for_input = False
        self.card_buttons = []  # For clickable cards

    def process_event(self, event):
        """Process a pygame event."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if any card was clicked
            for i, (rect, _) in enumerate(self.card_buttons):
                if rect.collidepoint(event.pos):
                    self.selected_index = i
                    self.waiting_for_input = False
                    return

    def show_dialog_buttons(self, options, start_y=400):
        """Display clickable buttons for options and wait for selection."""
        self.card_buttons = []
        button_width = 200
        button_height = 40
        spacing = 20

        # Calculate total width needed
        total_width = len(options) * (button_width + spacing) - spacing
        start_x = (self.ui.width - total_width) // 2

        # Create buttons
        for i, option in enumerate(options):
            x = start_x + i * (button_width + spacing)
            rect = pygame.Rect(x, start_y, button_width, button_height)
            self.card_buttons.append((rect, option))

        # Wait for selection
        self.waiting_for_input = True
        self.selected_index = None

        while self.waiting_for_input and self.ui.running:
            # Process events
            if not self.ui.process_events():
                return None

            # Draw background
            self.ui.screen.fill(BACKGROUND_COLOR)

            # Draw buttons
            for rect, text in self.card_buttons:
                # Draw button background
                pygame.draw.rect(self.ui.screen, (80, 80, 80), rect, border_radius=5)
                pygame.draw.rect(self.ui.screen, (120, 120, 120), rect, width=2, border_radius=5)

                # Draw button text
                self.ui.draw_text(text, self.ui.font_normal, (230, 230, 230),
                              rect.centerx, rect.centery, center=True)

            # Update display
            self.ui.update_display()
            pygame.time.wait(50)  # Reduce CPU usage

        # Return selected option
        if self.selected_index is not None and self.selected_index < len(options):
            return options[self.selected_index]
        return None

    # === InputHandler implementations ===

    def prompt_yes_no(self, prompt: str) -> bool:
        """Prompt the user for a yes/no answer."""
        self.ui.show_notification(prompt)
        result = self.show_dialog_buttons(["Yes", "No"])
        return result == "Yes"

    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player."""
        # Show a more visible and clear prompt
        title = f"Select Player {player_number} Type"

        # Display screen with options
        self.card_buttons = []
        options = [
            ("1: Human", "Play as a human player with manual card selection"),
            ("2: Random", "Random AI that makes completely random choices"),
            ("3: AI", "AI player that learns and adapts to the judge's preferences")
        ]

        option_width = 300
        option_height = 100
        spacing = 30
        start_y = 150

        # Create buttons
        for i, (option, description) in enumerate(options):
            y = start_y + i * (option_height + spacing)
            rect = pygame.Rect((self.ui.width - option_width) // 2, y, option_width, option_height)
            self.card_buttons.append((rect, (option, description)))

        # Display explanation text at top
        self.ui.show_notification(title)

        # Wait for selection
        self.waiting_for_input = True
        self.selected_index = None

        while self.waiting_for_input and self.ui.running:
            # Process events
            if not self.ui.process_events():
                return "2"  # Default to Random if window closed

            # Draw background
            self.ui.screen.fill(BACKGROUND_COLOR)

            # Draw title
            self.ui.draw_text(title, self.ui.font_large, (230, 230, 230),
                           self.ui.width // 2, 80, center=True)

            # Draw buttons
            for rect, (option, description) in self.card_buttons:
                # Draw button background
                pygame.draw.rect(self.ui.screen, (60, 60, 80), rect, border_radius=8)
                pygame.draw.rect(self.ui.screen, (100, 100, 140), rect, width=2, border_radius=8)

                # Draw button text
                self.ui.draw_text(option, self.ui.font_normal, (255, 255, 255),
                               rect.centerx, rect.centery - 15, center=True)
                self.ui.draw_text(description, self.ui.font_small, (200, 200, 200),
                               rect.centerx, rect.centery + 15, center=True)

            # Draw instructions at bottom
            self.ui.draw_text("Click on an option to select", self.ui.font_small, (180, 180, 180),
                           self.ui.width // 2, self.ui.height - 50, center=True)

            # Update display
            self.ui.update_display()
            pygame.time.wait(50)  # Reduce CPU usage

        # Return selected option or default
        if self.selected_index is not None and self.selected_index < len(options):
            return options[self.selected_index][0][0]  # Return just the number
        return "2"  # Default to Random

    def prompt_human_player_name(self) -> str:
        """Prompt for a human player's name."""
        title = "Enter Human Player Name"
        default_name = "Player"
        current_name = default_name
        cursor_visible = True
        cursor_time = 0

        # Draw the input box
        input_width = 300
        input_height = 50
        input_rect = pygame.Rect((self.ui.width - input_width) // 2, 200, input_width, input_height)

        # Initialize active state
        active = True
        done = False

        while not done and self.ui.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.ui.running = False
                    return default_name

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        current_name = current_name[:-1]
                    else:
                        # Only add printable characters
                        if event.unicode.isprintable() and len(current_name) < 20:
                            current_name += event.unicode

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if input_rect is clicked
                    active = input_rect.collidepoint(event.pos)

                    # Check for "Done" button
                    if done_rect.collidepoint(event.pos):
                        done = True

            # Draw background
            self.ui.screen.fill(BACKGROUND_COLOR)

            # Draw title
            self.ui.draw_text(title, self.ui.font_large, (230, 230, 230),
                           self.ui.width // 2, 100, center=True)

            # Draw input box
            box_color = (100, 100, 160) if active else (70, 70, 120)
            pygame.draw.rect(self.ui.screen, box_color, input_rect, border_radius=5)
            pygame.draw.rect(self.ui.screen, (150, 150, 200), input_rect, width=2, border_radius=5)

            # Update cursor visibility every 500ms
            if pygame.time.get_ticks() - cursor_time > 500:
                cursor_visible = not cursor_visible
                cursor_time = pygame.time.get_ticks()

            # Draw text with cursor
            display_text = current_name
            if active and cursor_visible:
                display_text += "|"

            self.ui.draw_text(display_text, self.ui.font_normal, (255, 255, 255),
                          input_rect.centerx, input_rect.centery, center=True)

            # Draw "Done" button
            done_rect = pygame.Rect((self.ui.width - 120) // 2, 300, 120, 40)
            pygame.draw.rect(self.ui.screen, (70, 120, 70), done_rect, border_radius=5)
            pygame.draw.rect(self.ui.screen, (100, 180, 100), done_rect, width=2, border_radius=5)
            self.ui.draw_text("Done", self.ui.font_normal, (255, 255, 255),
                          done_rect.centerx, done_rect.centery, center=True)

            # Update display
            self.ui.update_display()
            pygame.time.wait(50)  # Reduce CPU usage

        # Return the name (default if empty)
        return current_name if current_name else default_name

    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type."""
        self.ui.show_notification("Select AI model type")
        result = self.show_dialog_buttons(["1: Linear Regression", "2: Neural Network"])
        if result:
            return result[0]
        return "1"

    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype."""
        self.ui.show_notification("Select AI archetype")
        result = self.show_dialog_buttons(["1: Literalist", "2: Contrarian", "3: Comedian"])
        if result:
            return result[0]
        return "1"

    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the selection of the starting judge."""
        self.ui.show_notification("Select starting judge")
        options = [f"{i}" for i in range(1, player_count + 1)]
        result = self.show_dialog_buttons(options)
        if result:
            return int(result)
        return 1

    def prompt_human_agent_choose_red_apple(self, player: "Agent", red_apples: List["RedApple"],
                                         green_apple: "GreenApple") -> int:
        """Prompt a player to select a red apple."""
        self.ui.show_notification(f"Select a red apple to match: {green_apple.get_adjective()}")

        # Create card options
        options = [apple.get_noun() for apple in red_apples]

        # Show cards and wait for selection
        self.card_buttons = []
        card_width = 150
        card_height = 100
        spacing = 20

        # Calculate layout
        cards_per_row = min(5, len(options))
        total_width = cards_per_row * (card_width + spacing) - spacing
        start_x = (self.ui.width - total_width) // 2
        start_y = 300

        # Create card buttons
        for i, option in enumerate(options):
            row = i // cards_per_row
            col = i % cards_per_row
            x = start_x + col * (card_width + spacing)
            y = start_y + row * (card_height + spacing)
            rect = pygame.Rect(x, y, card_width, card_height)
            self.card_buttons.append((rect, option))

        # Wait for selection
        self.waiting_for_input = True
        self.selected_index = None

        while self.waiting_for_input and self.ui.running:
            # Process events
            if not self.ui.process_events():
                return 0

            # Draw background
            self.ui.screen.fill(BACKGROUND_COLOR)

            # Draw green apple
            green_text = f"Green Apple: {green_apple.get_adjective()}"
            self.ui.draw_text(green_text, self.ui.font_large, (100, 200, 100),
                          self.ui.width // 2, 150, center=True)

            # Draw prompt
            self.ui.draw_text(f"Select a red apple to match", self.ui.font_normal,
                          (230, 230, 230), self.ui.width // 2, 200, center=True)

            # Draw cards
            for rect, text in self.card_buttons:
                # Draw card background
                pygame.draw.rect(self.ui.screen, (80, 20, 20), rect, border_radius=5)
                pygame.draw.rect(self.ui.screen, (150, 50, 50), rect, width=2, border_radius=5)

                # Draw card text
                self.ui.draw_text(text, self.ui.font_normal, (230, 230, 230),
                              rect.centerx, rect.centery, center=True)

            # Update display
            self.ui.update_display()
            pygame.time.wait(50)  # Reduce CPU usage

        # Return selected index or 0 if none
        return self.selected_index if self.selected_index is not None else 0

    def prompt_judge_select_winner(self, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                                green_apple: "GreenApple") -> "Agent":
        """Prompt the judge to select the winning red apple."""
        self.ui.show_notification(f"Select the winning red apple for: {green_apple.get_adjective()}")

        # Create list of players and their submissions
        players = list(submissions.keys())

        # If no submissions, return the judge as a fallback (can't have no winner)
        if not players:
            logging.warning("No submissions to judge. Returning judge as winner.")
            return judge

        apples = [submissions[player] for player in players]

        # Show cards and wait for selection
        self.card_buttons = []
        card_width = 170
        card_height = 120
        spacing = 30

        # Calculate layout
        cards_per_row = min(4, len(apples))
        total_width = cards_per_row * (card_width + spacing) - spacing
        start_x = (self.ui.width - total_width) // 2
        start_y = 300

        # Create card buttons
        for i, (player, apple) in enumerate(zip(players, apples)):
            row = i // cards_per_row
            col = i % cards_per_row
            x = start_x + col * (card_width + spacing)
            y = start_y + row * (card_height + spacing)
            rect = pygame.Rect(x, y, card_width, card_height)
            self.card_buttons.append((rect, (apple.get_noun(), player.get_name())))

        # Wait for selection
        self.waiting_for_input = True
        self.selected_index = None

        while self.waiting_for_input and self.ui.running:
            # Process events
            if not self.ui.process_events():
                return players[0]

            # Draw background
            self.ui.screen.fill(BACKGROUND_COLOR)

            # Draw green apple
            green_text = f"Green Apple: {green_apple.get_adjective()}"
            self.ui.draw_text(green_text, self.ui.font_large, (100, 200, 100),
                          self.ui.width // 2, 150, center=True)

            # Draw prompt
            self.ui.draw_text(f"Judge {judge.get_name()}, select the winning card",
                          self.ui.font_normal, (230, 230, 230),
                          self.ui.width // 2, 200, center=True)

            # Draw cards
            for rect, (noun, player_name) in self.card_buttons:
                # Draw card background
                pygame.draw.rect(self.ui.screen, (80, 20, 20), rect, border_radius=5)
                pygame.draw.rect(self.ui.screen, (150, 50, 50), rect, width=2, border_radius=5)

                # Draw card text
                self.ui.draw_text(noun, self.ui.font_normal, (230, 230, 230),
                              rect.centerx, rect.centery - 15, center=True)
                self.ui.draw_text(f"({player_name})", self.ui.font_small, (200, 200, 200),
                              rect.centerx, rect.centery + 15, center=True)

            # Update display
            self.ui.update_display()
            pygame.time.wait(50)  # Reduce CPU usage

        # Return selected player or first if none
        if self.selected_index is not None and self.selected_index < len(players):
            return players[self.selected_index]
        return players[0]

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode."""
        self.ui.show_notification("Select training model type")
        result = self.show_dialog_buttons(["1: Linear Regression", "2: Neural Network"])
        if result:
            return result[0]
        return "1"

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode."""
        self.ui.show_notification("Select training pretrained type")
        result = self.show_dialog_buttons(["1: Literalist", "2: Contrarian", "3: Comedian"])
        if result:
            return result[0]
        return "1"
