# Description: Tkinter-based input handling for Apples to Apples game.

# Standard Libraries
from typing import List, Dict, TYPE_CHECKING
import logging

# Third-party Libraries
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
except ImportError:
    logging.error("Tkinter not installed. Install with: pip install tkinter")
    raise ImportError("Tkinter is required for this UI. Install with: pip install tkinter")

# Local Modules
from src.interface.input.input_handler import InputHandler
from src.ui.gui.tkinter.tkinter_dialogs import (
    PlayerTypeDialog, ModelTypeDialog, ArchetypeDialog, StartingJudgeDialog,
    RedAppleSelectionDialog, JudgeSelectionDialog, TrainingModelTypeDialog,
    TrainingPretrainedTypeDialog
)

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.ui.gui.tkinter.tkinter_ui import TkinterUI


class TkinterInputHandler(InputHandler):
    """Implementation of InputHandler for Tkinter GUI interface."""

    def __init__(self, root: tk.Tk, ui: "TkinterUI"):
        """
        Initialize the Tkinter input handler.

        Args:
            root: The Tkinter root window
            ui: The TkinterUI instance
        """
        self.root = root
        self.ui = ui

    def prompt_yes_no(self, prompt: str) -> bool:
        """Prompt the user for a yes/no answer via dialog box."""
        return messagebox.askyesno("Question", prompt)

    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player via dialog box."""
        dialog = PlayerTypeDialog(self.root, player_number)
        return dialog.result

    def prompt_human_player_name(self) -> str:
        """Prompt for a human player's name via dialog box."""
        name = simpledialog.askstring("Input", "Please enter your name:", parent=self.root)
        while not name:
            messagebox.showerror("Error", "Name cannot be empty.")
            name = simpledialog.askstring("Input", "Please enter your name:", parent=self.root)
        return name

    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type via dialog box."""
        dialog = ModelTypeDialog(self.root)
        return dialog.result

    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype via dialog box."""
        dialog = ArchetypeDialog(self.root)
        return dialog.result

    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the selection of the starting judge via dialog box."""
        dialog = StartingJudgeDialog(self.root, player_count)
        return dialog.result

    def prompt_human_agent_choose_red_apple(self, player: "Agent", red_apples: List["RedApple"],
                       green_apple: "GreenApple") -> int:
        """Prompt a player to select a red apple."""
        # Reset any previous selection
        self.ui.selected_card_index = None

        # Display the player's cards first
        self.ui.output_handler.display_player_red_apples(player)

        # Create a variable to track if selection is complete
        selection_complete = False

        # Define a callback for the card selection
        def on_selection_complete():
            nonlocal selection_complete
            selection_complete = True

        # Set the confirmation callback - will be called directly when a card is clicked
        self.ui.on_card_confirm = on_selection_complete

        # Wait for user to select a card
        while not selection_complete:
            self.root.update()

        # Get the final selection
        index = self.ui.selected_card_index

        # If somehow we don't have a selection, default to first card
        if index is None:
            index = 0

        return index

    def prompt_judge_select_winner(self, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                              green_apple: "GreenApple") -> "Agent":
        """Prompt the judge to select the winning red apple via dialog box."""
        try:
            logging.debug(f"Opening JudgeSelectionDialog for {judge.get_name()}")
            dialog = JudgeSelectionDialog(self.root, judge, submissions, green_apple)
            selected_player = dialog.result
            logging.debug(f"Judge {judge.get_name()} selected: {selected_player.get_name()}")

            # Make sure the root window gets focus after dialog closes
            self.root.focus_force()
            self.root.update()

            return selected_player
        except Exception as e:
            logging.error(f"Error in prompt_judge_select_winner: {e}")
            # Return first player as fallback
            return list(submissions.keys())[0]

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode via dialog box."""
        dialog = TrainingModelTypeDialog(self.root)
        return dialog.result

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode via dialog box."""
        dialog = TrainingPretrainedTypeDialog(self.root)
        return dialog.result
