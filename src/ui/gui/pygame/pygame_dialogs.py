# Description: Pygame dialog classes for user interaction in the game.

# Standard Libraries
import pygame
import logging
from typing import Optional, List, Dict, Tuple, Any, Callable, TYPE_CHECKING

# Local module imports
from src.ui.gui.pygame.pygame_ui import BACKGROUND_COLOR

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple


class Dialog:
    """Base class for Pygame dialogs."""

    def __init__(self, screen, title: str):
        """Initialize a dialog with a title."""
        self.screen = screen
        self.title = title
        self.width, self.height = screen.get_size()
        self.running = True
        self.result = None

        # Initialize fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 32)
        self.font_normal = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Define common colors
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_BUTTON = (80, 80, 100)
        self.COLOR_BUTTON_HIGHLIGHT = (100, 100, 140)
        self.COLOR_BUTTON_TEXT = (255, 255, 255)
        self.COLOR_HEADING = (180, 180, 255)

    def draw_text(self, text: str, font, color, x: int, y: int, center: bool = False):
        """Helper function to draw text on the screen."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()

        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)

        self.screen.blit(text_surface, text_rect)
        return text_rect

    def draw_button(self, rect, text: str, color=None, highlight: bool = False) -> pygame.Rect:
        """Draw a button with text."""
        if color is None:
            color = self.COLOR_BUTTON

        # Draw button background with highlight if needed
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        if highlight:
            pygame.draw.rect(self.screen, self.COLOR_BUTTON_HIGHLIGHT, rect, width=3, border_radius=8)
        else:
            pygame.draw.rect(self.screen, self.COLOR_BUTTON_HIGHLIGHT, rect, width=1, border_radius=8)

        # Draw button text
        self.draw_text(text, self.font_normal, self.COLOR_BUTTON_TEXT,
                      rect.centerx, rect.centery, center=True)

        return rect

    def run(self) -> Any:
        """Run the dialog and return the result."""
        clock = pygame.time.Clock()

        while self.running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                # Handle mouse events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(event.pos)

            # Draw dialog background
            self.screen.fill(BACKGROUND_COLOR)

            # Draw dialog content
            self.draw()

            # Update display
            pygame.display.flip()
            clock.tick(30)

        return self.result

    def draw(self):
        """Draw the dialog content (to be implemented by subclasses)."""
        # Draw title
        self.draw_text(self.title, self.font_title, self.COLOR_HEADING,
                     self.width // 2, 50, center=True)

    def on_click(self, pos):
        """Handle mouse clicks (to be implemented by subclasses)."""
        pass

    def close(self, result):
        """Close the dialog with the specified result."""
        self.result = result
        self.running = False


class PlayerTypeDialog(Dialog):
    """Dialog for selecting a player type."""

    def __init__(self, screen, player_number: int):
        """Initialize dialog for selecting player type."""
        super().__init__(screen, f"Select Type for Player {player_number}")
        self.player_number = player_number
        self.options = [
            ("1", "Human", "Play as a human player with manual card selection"),
            ("2", "Random", "AI that makes completely random choices"),
            ("3", "AI", "AI player that learns and adapts to the judge")
        ]
        self.option_rects = []

    def draw(self):
        """Draw the player type selection dialog."""
        super().draw()

        # Draw instructions
        self.draw_text("Click on an option to select player type:",
                     self.font_normal, self.COLOR_TEXT,
                     self.width // 2, 100, center=True)

        # Reset option rectangles
        self.option_rects = []

        # Draw option buttons
        option_width = 300
        option_height = 80
        spacing = 20
        start_y = 150

        for i, (value, label, description) in enumerate(self.options):
            rect = pygame.Rect(
                (self.width - option_width) // 2,
                start_y + i * (option_height + spacing),
                option_width,
                option_height
            )

            # Draw the button
            self.draw_button(rect, f"{label}")

            # Draw the description under the main text
            self.draw_text(description, self.font_small, (200, 200, 200),
                         rect.centerx, rect.centery + 15, center=True)

            # Store the rectangle for click detection
            self.option_rects.append((rect, value))

    def on_click(self, pos):
        """Handle mouse clicks on options."""
        for rect, value in self.option_rects:
            if rect.collidepoint(pos):
                self.close(value)
                break


class NameInputDialog(Dialog):
    """Dialog for entering a player name."""

    def __init__(self, screen):
        """Initialize dialog for entering player name."""
        super().__init__(screen, "Enter Human Player Name")
        self.current_name = "Player"
        self.cursor_visible = True
        self.cursor_time = 0
        self.done_rect = None

    def draw(self):
        """Draw the name input dialog."""
        super().draw()

        # Draw instructions
        self.draw_text("Enter your name:", self.font_normal, self.COLOR_TEXT,
                     self.width // 2, 100, center=True)

        # Draw input box
        input_width = 300
        input_height = 50
        input_rect = pygame.Rect(
            (self.width - input_width) // 2,
            150,
            input_width,
            input_height
        )

        # Draw input background
        pygame.draw.rect(self.screen, (70, 70, 100), input_rect, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 140), input_rect, width=2, border_radius=5)

        # Update cursor visibility every 500ms
        if pygame.time.get_ticks() - self.cursor_time > 500:
            self.cursor_visible = not self.cursor_visible
            self.cursor_time = pygame.time.get_ticks()

        # Draw name with cursor
        display_text = self.current_name
        if self.cursor_visible:
            display_text += "|"

        self.draw_text(display_text, self.font_normal, (255, 255, 255),
                     input_rect.centerx, input_rect.centery, center=True)

        # Draw Done button
        done_rect = pygame.Rect(
            (self.width - 100) // 2,
            220,
            100,
            40
        )
        self.done_rect = self.draw_button(done_rect, "Done")

    def on_click(self, pos):
        """Handle mouse clicks."""
        if self.done_rect and self.done_rect.collidepoint(pos):
            self.close(self.current_name)

    def run(self) -> str:
        """Run the dialog with keyboard input support."""
        clock = pygame.time.Clock()

        while self.running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return "Player"  # Default if closed

                # Handle keyboard events
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.close(self.current_name)
                    elif event.key == pygame.K_BACKSPACE:
                        self.current_name = self.current_name[:-1]
                    else:
                        # Only add printable characters
                        if event.unicode.isprintable() and len(self.current_name) < 20:
                            self.current_name += event.unicode

                # Handle mouse events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(event.pos)

            # Draw dialog
            self.screen.fill(BACKGROUND_COLOR)
            self.draw()

            # Update display
            pygame.display.flip()
            clock.tick(30)

        return self.result or "Player"  # Default if somehow no result


class ModelTypeDialog(Dialog):
    """Dialog for selecting an AI model type."""

    def __init__(self, screen):
        """Initialize dialog for selecting AI model type."""
        super().__init__(screen, "Select AI Model Type")
        self.options = [
            ("1", "Linear Regression", "Simple, efficient model for learning patterns"),
            ("2", "Neural Network", "More complex model with potentially better results")
        ]
        self.option_rects = []

    def draw(self):
        """Draw the model type selection dialog."""
        super().draw()

        # Draw instructions
        self.draw_text("Select the machine learning model to use:",
                     self.font_normal, self.COLOR_TEXT,
                     self.width // 2, 100, center=True)

        # Reset option rectangles
        self.option_rects = []

        # Draw option buttons
        option_width = 300
        option_height = 80
        spacing = 20
        start_y = 150

        for i, (value, label, description) in enumerate(self.options):
            rect = pygame.Rect(
                (self.width - option_width) // 2,
                start_y + i * (option_height + spacing),
                option_width,
                option_height
            )

            # Draw the button
            self.draw_button(rect, f"{label}")

            # Draw the description under the main text
            self.draw_text(description, self.font_small, (200, 200, 200),
                         rect.centerx, rect.centery + 15, center=True)

            # Store the rectangle for click detection
            self.option_rects.append((rect, value))

    def on_click(self, pos):
        """Handle mouse clicks on options."""
        for rect, value in self.option_rects:
            if rect.collidepoint(pos):
                self.close(value)
                break


class ArchetypeDialog(Dialog):
    """Dialog for selecting an AI archetype."""

    def __init__(self, screen):
        """Initialize dialog for selecting AI archetype."""
        super().__init__(screen, "Select AI Archetype")
        self.options = [
            ("1", "Literalist", "Chooses cards based on literal definitions"),
            ("2", "Contrarian", "Tends to pick unusual or unexpected matches"),
            ("3", "Comedian", "Prioritizes funny or ironic card combinations")
        ]
        self.option_rects = []

    def draw(self):
        """Draw the archetype selection dialog."""
        super().draw()

        # Draw instructions
        self.draw_text("Select the AI personality:",
                     self.font_normal, self.COLOR_TEXT,
                     self.width // 2, 100, center=True)

        # Reset option rectangles
        self.option_rects = []

        # Draw option buttons
        option_width = 300
        option_height = 80
        spacing = 20
        start_y = 150

        for i, (value, label, description) in enumerate(self.options):
            rect = pygame.Rect(
                (self.width - option_width) // 2,
                start_y + i * (option_height + spacing),
                option_width,
                option_height
            )

            # Draw the button
            self.draw_button(rect, f"{label}")

            # Draw the description under the main text
            self.draw_text(description, self.font_small, (200, 200, 200),
                         rect.centerx, rect.centery + 15, center=True)

            # Store the rectangle for click detection
            self.option_rects.append((rect, value))

    def on_click(self, pos):
        """Handle mouse clicks on options."""
        for rect, value in self.option_rects:
            if rect.collidepoint(pos):
                self.close(value)
                break


class JudgeSelectionDialog(Dialog):
    """Dialog for selecting the winning red apple."""

    def __init__(self, screen, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                green_apple: "GreenApple"):
        """Initialize dialog for judge selection."""
        super().__init__(screen, f"{judge.get_name()} selects the winner")
        self.judge = judge
        self.green_apple = green_apple
        self.players = list(submissions.keys())
        self.apples = [submissions[player] for player in self.players]
        self.card_rects = []

    def draw(self):
        """Draw the judge selection dialog."""
        super().draw()

        # Draw green apple
        green_text = f"Green Apple: {self.green_apple.get_adjective()}"
        self.draw_text(green_text, self.font_normal, (100, 200, 100),
                     self.width // 2, 100, center=True)

        # Draw instructions
        self.draw_text(f"Click on the best match for '{self.green_apple.get_adjective()}':",
                     self.font_normal, self.COLOR_TEXT,
                     self.width // 2, 140, center=True)

        # Reset card rectangles
        self.card_rects = []

        # Draw red apple cards
        card_width = 180
        card_height = 150
        spacing = 20
        cards_per_row = min(4, len(self.apples))
        total_width = cards_per_row * (card_width + spacing) - spacing
        start_x = (self.width - total_width) // 2
        start_y = 180

        for i, (player, apple) in enumerate(zip(self.players, self.apples)):
            row = i // cards_per_row
            col = i % cards_per_row
            x = start_x + col * (card_width + spacing)
            y = start_y + row * (card_height + spacing) + 20

            # Draw card background
            card_rect = pygame.Rect(x, y, card_width, card_height)
            pygame.draw.rect(self.screen, (80, 20, 20), card_rect, border_radius=8)
            pygame.draw.rect(self.screen, (150, 50, 50), card_rect, width=2, border_radius=8)

            # Draw card title
            self.draw_text("RED APPLE", self.font_small, (200, 100, 100),
                         card_rect.centerx, y + 20, center=True)

            # Draw noun
            self.draw_text(apple.get_noun(), self.font_normal, (255, 200, 200),
                         card_rect.centerx, card_rect.centery - 15, center=True)

            # Draw player name
            self.draw_text(f"({player.get_name()})", self.font_small, (200, 150, 150),
                         card_rect.centerx, card_rect.centery + 30, center=True)

            # Store card rectangle and associated player
            self.card_rects.append((card_rect, player))

    def on_click(self, pos):
        """Handle mouse clicks on red apple cards."""
        for rect, player in self.card_rects:
            if rect.collidepoint(pos):
                self.close(player)
                break


class TrainingModelTypeDialog(ModelTypeDialog):
    """Dialog for selecting training model type."""

    def __init__(self, screen):
        super().__init__(screen)
        self.title = "Select Training Model Type"


class TrainingPretrainedTypeDialog(ArchetypeDialog):
    """Dialog for selecting training pretrained type."""

    def __init__(self, screen):
        super().__init__(screen)
        self.title = "Select Training Pretrained Type"


class RedCardSelectionDialog(Dialog):
    """Dialog for human player to select a red card from their hand."""

    def __init__(self, screen, player: "Agent", red_apples: List["RedApple"],
                green_apple: "GreenApple"):
        """Initialize dialog for red card selection."""
        super().__init__(screen, f"{player.get_name()}, Select a Red Apple")
        self.player = player
        self.red_apples = red_apples
        self.green_apple = green_apple
        self.card_rects = []

    def draw(self):
        """Draw the red card selection dialog."""
        super().draw()

        # Draw green apple at the top
        green_text = f"Green Apple: {self.green_apple.get_adjective()}"
        self.draw_text(green_text, self.font_large, (100, 200, 100),
                     self.width // 2, 100, center=True)

        # Draw instructions
        self.draw_text(f"Select a red apple that best matches '{self.green_apple.get_adjective()}':",
                     self.font_normal, self.COLOR_TEXT,
                     self.width // 2, 140, center=True)

        # Reset card rectangles
        self.card_rects = []

        # Draw red apple cards
        card_width = 160
        card_height = 140
        spacing = 20
        cards_per_row = min(4, len(self.red_apples))

        # Calculate layout
        total_width = cards_per_row * (card_width + spacing) - spacing
        start_x = (self.width - total_width) // 2
        start_y = 180

        for i, apple in enumerate(self.red_apples):
            row = i // cards_per_row
            col = i % cards_per_row
            x = start_x + col * (card_width + spacing)
            y = start_y + row * (card_height + spacing) + 20

            # Draw card background
            card_rect = pygame.Rect(x, y, card_width, card_height)
            pygame.draw.rect(self.screen, (80, 20, 20), card_rect, border_radius=8)
            pygame.draw.rect(self.screen, (150, 50, 50), card_rect, width=2, border_radius=8)

            # Draw card title
            self.draw_text("RED APPLE", self.font_small, (200, 100, 100),
                         card_rect.centerx, y + 20, center=True)

            # Draw noun
            self.draw_text(apple.get_noun(), self.font_normal, (255, 200, 200),
                         card_rect.centerx, card_rect.centery - 20, center=True)

            # Draw description (truncated)
            if hasattr(apple, "get_description"):
                description = apple.get_description()
                if description:
                    if len(description) > 25:
                        description = description[:22] + "..."
                    self.draw_text(description, self.font_small, (200, 150, 150),
                                 card_rect.centerx, card_rect.centery + 15, center=True)

            # Store card rectangle and associated apple
            self.card_rects.append((card_rect, i))  # Store index instead of apple

    def on_click(self, pos):
        """Handle mouse clicks on red apple cards."""
        for rect, index in self.card_rects:
            if rect.collidepoint(pos):
                self.close(index)
                break
