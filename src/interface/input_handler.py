# Description: Abstract input handling for Apples to Apples game. Terminal-based and Tkinter-based implementations.

# Standard Libraries
from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING

# Third-party Libraries
import tkinter as tk
from tkinter import simpledialog, messagebox

# Local Modules
from src.ui.gui.tkinter_dialogs import (
    PlayerTypeDialog, ModelTypeDialog, ArchetypeDialog, StartingJudgeDialog,
    RedAppleSelectionDialog, JudgeSelectionDialog, TrainingModelTypeDialog,
    TrainingPretrainedTypeDialog
)

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.ui.gui.tkinter_ui import TkinterUI


class InputHandler(ABC):
    """Abstract class for handling user input in various UIs."""

    @abstractmethod
    def prompt_yes_no(self, prompt: str) -> bool:
        """
        Prompt the user for a yes/no answer.

        Args:
            prompt: The question to ask the user

        Returns:
            True for yes, False for no
        """
        pass

    @abstractmethod
    def prompt_player_type(self, player_number: int) -> str:
        """
        Prompt for the type of player.

        Args:
            player_number: The player number (1-based)

        Returns:
            '1' for Human, '2' for Random, '3' for AI
        """
        pass

    @abstractmethod
    def prompt_human_player_name(self) -> str:
        """
        Prompt for a human player's name.

        Returns:
            The player's name
        """
        pass

    @abstractmethod
    def prompt_ai_model_type(self) -> str:
        """
        Prompt for the AI model type.

        Returns:
            '1' for Linear Regression, '2' for Neural Network
        """
        pass

    @abstractmethod
    def prompt_ai_archetype(self) -> str:
        """
        Prompt for the AI archetype.

        Returns:
            '1' for Literalist, '2' for Contrarian, '3' for Comedian
        """
        pass

    @abstractmethod
    def prompt_starting_judge(self, player_count: int) -> int:
        """
        Prompt for the selection of the starting judge.

        Args:
            player_count: The number of players

        Returns:
            1-based index of the selected judge
        """
        pass

    @abstractmethod
    def prompt_human_agent_choose_red_apple(self, player: "Agent", red_apples: List["RedApple"],
                               green_apple: "GreenApple") -> int:
        """
        Prompt a player to select a red apple.

        Args:
            player: The player selecting the card
            red_apples: List of red apples to choose from
            green_apple: The green apple in play

        Returns:
            The index of the selected red apple
        """
        pass

    @abstractmethod
    def prompt_judge_select_winner(self, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                                 green_apple: "GreenApple") -> "Agent":
        """
        Prompt the judge to select the winning red apple.

        Args:
            judge: The judge making the selection
            submissions: Dictionary mapping players to their submitted red apples
            green_apple: The green apple in play

        Returns:
            The player whose red apple was selected as the winner
        """
        pass

    @abstractmethod
    def prompt_training_model_type(self) -> str:
        """
        Prompt for the model type in training mode.

        Returns:
            '1' for Linear Regression, '2' for Neural Network
        """
        pass

    @abstractmethod
    def prompt_training_pretrained_type(self) -> str:
        """
        Prompt for the pretrained model type in training mode.

        Returns:
            '1' for Literalist, '2' for Contrarian, '3' for Comedian
        """
        pass


class TerminalInputHandler(InputHandler):
    """Implementation of InputHandler for terminal/console interface."""

    def __init__(self, print_in_terminal: bool = True):
        """Initialize the terminal input handler."""
        self.print_in_terminal = print_in_terminal

    def prompt_yes_no(self, prompt: str) -> bool:
        """Prompt the user for a yes/no answer via terminal input."""
        while True:
            response = input(f"{prompt} (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            print("Invalid input. Please enter 'y' or 'n'.")

    def prompt_player_type(self, player_number: int) -> str:
        """Prompt for the type of player via terminal input."""
        if self.print_in_terminal:
            print(f"\nWhat type is Agent {player_number}?")

        while True:
            player_type = input("Please enter the player type (1: Human, 2: Random, 3: AI): ")
            if player_type in ['1', '2', '3']:
                return player_type
            print("Invalid input. Please enter '1', '2', or '3'.")

    def prompt_human_player_name(self) -> str:
        """Prompt for a human player's name via terminal input."""
        while True:
            name = input("Please enter your name: ").strip()
            if name:
                return name
            print("Name cannot be empty. Please enter a valid name.")

    def prompt_ai_model_type(self) -> str:
        """Prompt for the AI model type via terminal input."""
        while True:
            model_type = input("Please enter the machine learning model (1: Linear Regression, 2: Neural Network): ")
            if model_type in ['1', '2']:
                return model_type
            print("Invalid input. Please enter '1' or '2'.")

    def prompt_ai_archetype(self) -> str:
        """Prompt for the AI archetype via terminal input."""
        while True:
            archetype = input("Please enter the pretrained archetype (1: Literalist, 2: Contrarian, 3: Comedian): ")
            if archetype in ['1', '2', '3']:
                return archetype
            print("Invalid input. Please enter '1', '2', or '3'.")

    def prompt_starting_judge(self, player_count: int) -> int:
        """Prompt for the selection of the starting judge via terminal input."""
        while True:
            try:
                choice = input(f"\nPlease choose the starting judge (1-{player_count}): ")
                judge_index = int(choice)
                if 1 <= judge_index <= player_count:
                    return judge_index
                print(f"Invalid input. Please enter a number between 1 and {player_count}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def prompt_human_agent_choose_red_apple(self, player: "Agent", red_apples: List["RedApple"],
                              green_apple: "GreenApple") -> int:
        """Prompt a player to select a red apple via terminal input."""
        if self.print_in_terminal:
            print(f"\nGreen Apple: {green_apple.get_adjective()} ({green_apple.get_synonyms()})")
            print(f"\n{player.get_name()}, please select a red apple:")

            for i, apple in enumerate(red_apples):
                print(f"{i+1}. {apple.get_noun()} ({apple.get_description()})")

        # Prompt the human agent to choose a red apple
        red_apple_len = len(red_apples)
        red_apple_index = input(f"Choose a red apple (1 - {red_apple_len}): ")

        # Validate the input
        while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
            print(f"Invalid input. Please choose a valid red apple (1 - {red_apple_len}).")
            red_apple_index = input("Choose a red apple: ")

        # Convert the input to an index
        red_apple_index = int(red_apple_index) - 1
        return red_apple_index

    def prompt_judge_select_winner(self, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                                green_apple: "GreenApple") -> "Agent":
        """Prompt the judge to select the winning red apple via terminal input."""
        if self.print_in_terminal:
            print(f"\nGreen Apple: {green_apple.get_adjective()} ({green_apple.get_synonyms()})")
            print(f"\n{judge.get_name()}, please select the winning red apple:")

            players = list(submissions.keys())
            for i, player in enumerate(players):
                apple = submissions[player]
                print(f"{i+1}. {apple.get_noun()} ({apple.get_description()})")

        while True:
            try:
                choice = input(f"Enter your choice (1-{len(submissions)}): ")
                index = int(choice) - 1  # Convert to 0-based index
                if 0 <= index < len(submissions):
                    return list(submissions.keys())[index]
                print(f"Invalid input. Please enter a number between 1 and {len(submissions)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode via terminal input."""
        while True:
            model_type = input("Please enter the model type (1: Linear Regression, 2: Neural Network): ")
            if model_type in ['1', '2']:
                return model_type
            print("Invalid input. Please enter '1' or '2'.")

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode via terminal input."""
        while True:
            pretrained_type = input("Please enter the pretrained model type (1: Literalist, 2: Contrarian, 3: Comedian): ")
            if pretrained_type in ['1', '2', '3']:
                return pretrained_type
            print("Invalid input. Please enter '1', '2', or '3'.")


class TkinterInputHandler(InputHandler):
    """Implementation of InputHandler for Tkinter GUI interface."""

    def __init__(self, root: tk.Tk, ui: 'TkinterUI'):
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
        """Prompt a player to select a red apple via dialog box."""
        dialog = RedAppleSelectionDialog(self.root, player, red_apples, green_apple)
        return dialog.result

    def prompt_judge_select_winner(self, judge: "Agent", submissions: Dict["Agent", "RedApple"],
                                 green_apple: "GreenApple") -> "Agent":
        """Prompt the judge to select the winning red apple via dialog box."""
        dialog = JudgeSelectionDialog(self.root, judge, submissions, green_apple)
        return dialog.result

    def prompt_training_model_type(self) -> str:
        """Prompt for the model type in training mode via dialog box."""
        dialog = TrainingModelTypeDialog(self.root)
        return dialog.result

    def prompt_training_pretrained_type(self) -> str:
        """Prompt for the pretrained model type in training mode via dialog box."""
        dialog = TrainingPretrainedTypeDialog(self.root)
        return dialog.result
