# Description: CustomTkinter-based input handling for Apples to Apples game.

# Standard Libraries
from typing import List, Dict, TYPE_CHECKING
import logging

# Third-party Libraries
try:
    import customtkinter as ctk
except ImportError:
    logging.error("CustomTkinter not installed. Install with: pip install customtkinter")
    raise ImportError("CustomTkinter is required for this UI. Install with: pip install customtkinter")

# Local Modules
from src.interface.input.input_handler import InputHandler

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple


class CustomTkinterInputHandler(InputHandler):
    """CustomTkinter implementation of InputHandler."""

    def __init__(self, root: ctk.CTk):
        """Initialize the CustomTkinter input handler."""
        self.root = root

    def prompt_yes_no(self, prompt: str) -> bool:
        """Prompt the user for a yes/no answer via dialog box."""
        dialog = CTkMessageBox(self.root, title="Question", message=prompt,
                               option_1="Yes", option_2="No")
        return dialog.get() == "Yes"

    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player via dialog box."""
        dialog = PlayerTypeDialog(self.root, player_number)
        return dialog.result if dialog.result else '2'  # Default to Random if None

    def prompt_human_player_name(self) -> str:
        """Prompt for a human player's name via dialog box."""
        dialog = InputDialog(self.root, "Player Name", "Enter your name:")
        name = dialog.get_input()
        while not name:
            error_dialog = CTkMessageBox(self.root, title="Error", message="Name cannot be empty.",
                                         icon="cancel")
            error_dialog.get()
            dialog = InputDialog(self.root, "Player Name", "Enter your name:")
            name = dialog.get_input()
        return name

    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type via dialog box."""
        dialog = ModelTypeDialog(self.root)
        return dialog.result if dialog.result else '1'  # Default to Linear Regression

    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype via dialog box."""
        dialog = ArchetypeDialog(self.root)
        return dialog.result if dialog.result else '1'  # Default to Literalist

    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the selection of the starting judge via dialog box."""
        dialog = StartingJudgeDialog(self.root, player_count)
        return dialog.result if dialog.result else 1  # Default to first player

    def prompt_human_agent_choose_red_apple(self, player: "Agent", red_apples: List["RedApple"],
                                           green_apple: "GreenApple") -> int:
        """Prompt a player to select a red apple. This method should be implemented in the UI class."""
        # This is a placeholder - the actual implementation will be in CustomTkinterUI
        dialog = RedAppleSelectionDialog(self.root, player, red_apples, green_apple)
        return dialog.result if dialog.result is not None else 0

    def prompt_judge_select_winner(self, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                                  green_apple: "GreenApple") -> "Agent":
        """Prompt the judge to select the winning red apple via dialog box."""
        dialog = JudgeSelectionDialog(self.root, judge, submissions, green_apple)
        return dialog.result if dialog.result else list(submissions.keys())[0]  # Default to first player

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode via dialog box."""
        dialog = TrainingModelTypeDialog(self.root)
        return dialog.result if dialog.result else '1'  # Default to Linear Regression

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode via dialog box."""
        dialog = TrainingPretrainedTypeDialog(self.root)
        return dialog.result if dialog.result else '1'  # Default to Literalist
