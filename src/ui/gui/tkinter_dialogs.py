# Description: Dialog boxes for Tkinter UI in Apples to Apples.

# Standard Libraries
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, TYPE_CHECKING
from src.ui.gui.tkinter_widgets import GreenAppleCard, RedAppleCard

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple


class PlayerTypeDialog(tk.Toplevel):
    """Dialog for selecting player type."""
    def __init__(self, parent, player_number):
        super().__init__(parent)
        self.title(f"Player {player_number} Type")
        self.result = '2'  # Default to Random
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"What type is Player {player_number}?").grid(row=0, column=0, columnspan=3, pady=10)

        ttk.Button(frame, text="Human", command=lambda: self.set_result('1')).grid(row=1, column=0, padx=5, pady=10)
        ttk.Button(frame, text="Random", command=lambda: self.set_result('2')).grid(row=1, column=1, padx=5, pady=10)
        ttk.Button(frame, text="AI", command=lambda: self.set_result('3')).grid(row=1, column=2, padx=5, pady=10)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = '2'  # Default to Random if dialog is closed
        self.destroy()


class ModelTypeDialog(tk.Toplevel):
    """Dialog for selecting ML model type."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("AI Model Type")
        self.result = '1' # Default to Linear Regression
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Please select the machine learning model:").grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Linear Regression", command=lambda: self.set_result('1')).grid(row=1, column=0, padx=5, pady=10)
        ttk.Button(frame, text="Neural Network", command=lambda: self.set_result('2')).grid(row=1, column=1, padx=5, pady=10)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = '1'  # Default to Linear Regression if dialog is closed
        self.destroy()


class ArchetypeDialog(tk.Toplevel):
    """Dialog for selecting AI archetype."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("AI Archetype")
        self.result = '1'  # Default to Literalist
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Please select the AI archetype:").grid(row=0, column=0, columnspan=3, pady=10)

        ttk.Button(frame, text="Literalist", command=lambda: self.set_result('1')).grid(row=1, column=0, padx=5, pady=10)
        ttk.Button(frame, text="Contrarian", command=lambda: self.set_result('2')).grid(row=1, column=1, padx=5, pady=10)
        ttk.Button(frame, text="Comedian", command=lambda: self.set_result('3')).grid(row=1, column=2, padx=5, pady=10)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = '1'  # Default to Literalist if dialog is closed
        self.destroy()


class StartingJudgeDialog(tk.Toplevel):
    """Dialog for selecting the starting judge."""
    def __init__(self, parent, player_count):
        super().__init__(parent)
        self.title("Starting Judge Selection")
        self.result = 1  # Default to Player 1
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Please select the starting judge:").grid(row=0, column=0, columnspan=player_count, pady=10)

        for i in range(player_count):
            ttk.Button(frame, text=f"Player {i+1}", command=lambda j=i+1: self.set_result(j)).grid(
                row=1, column=i, padx=5, pady=10)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = 1  # Default to Player 1 if dialog is closed
        self.destroy()


class RedAppleSelectionDialog(tk.Toplevel):
    """Dialog for selecting a red apple card."""
    def __init__(self, parent, player, red_apples, green_apple):
        super().__init__(parent)
        self.title(f"{player.get_name()}'s Red Apple Selection")
        self.result = 0  # Default to first card
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.red_apples = red_apples

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Display green apple at the top
        green_card_frame = ttk.Frame(frame)
        green_card_frame.grid(row=0, column=0, columnspan=2, pady=10)
        green_card = GreenAppleCard.from_green_apple(green_card_frame, green_apple)
        green_card.pack(pady=5)

        ttk.Label(frame, text=f"{player.get_name()}, please select a red apple:").grid(
            row=1, column=0, columnspan=2, pady=10)

        card_frame = ttk.Frame(frame)
        card_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Create red apple cards using the new RedAppleCard class
        for i, apple in enumerate(red_apples):
            card = RedAppleCard.from_red_apple(
                card_frame,
                apple,
                command=lambda index=i: self.set_result(index)
            )
            card.grid(row=i//3, column=i%3, padx=5, pady=5)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = 0  # Default to first card if dialog is closed
        self.destroy()


class JudgeSelectionDialog(tk.Toplevel):
    """Dialog for the judge to select the winning red apple."""
    def __init__(self, parent, judge, submissions, green_apple):
        super().__init__(parent)
        self.title(f"{judge.get_name()}'s Judge Selection")
        self.players = list(submissions.keys())
        self.result = self.players[0] # Default to first player
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.submissions = submissions

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Display green apple at the top
        green_card_frame = ttk.Frame(frame)
        green_card_frame.grid(row=0, column=0, columnspan=len(self.players), pady=10)
        green_card = GreenAppleCard.from_green_apple(green_card_frame, green_apple)
        green_card.pack(pady=5)

        ttk.Label(frame, text=f"{judge.get_name()}, please select the winning red apple:").grid(
            row=1, column=0, columnspan=len(self.players), pady=10)

        card_frame = ttk.Frame(frame)
        card_frame.grid(row=2, column=0, columnspan=len(self.players), pady=10)

        # Create red apple cards using the new RedAppleCard class
        for i, player in enumerate(self.players):
            apple = submissions[player]

            card = RedAppleCard.from_red_apple(
                card_frame,
                apple,
                command=lambda index=i: self.set_result(index)
            )
            card.grid(row=i//3, column=i%3, padx=5, pady=5)

            # Add player name label below card
            player_label = ttk.Label(card_frame, text=player.get_name())
            player_label.grid(row=(i//3)+1, column=i%3)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = self.players[value]
        self.destroy()

    def cancel(self):
        self.result = self.players[0]  # Default to first player if dialog is closed
        self.destroy()


class TrainingModelTypeDialog(tk.Toplevel):
    """Dialog for selecting training model type."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Training Model Type")
        self.result = '1'  # Default to Linear Regression
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Please select the training model type:").grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="Linear Regression", command=lambda: self.set_result('1')).grid(row=1, column=0, padx=5, pady=10)
        ttk.Button(frame, text="Neural Network", command=lambda: self.set_result('2')).grid(row=1, column=1, padx=5, pady=10)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = '1'  # Default to Linear Regression if dialog is closed
        self.destroy()


class TrainingPretrainedTypeDialog(tk.Toplevel):
    """Dialog for selecting training pretrained type."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Training Pretrained Type")
        self.result = '3'  # Default to Comedian
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Please select the pretrained model type:").grid(row=0, column=0, columnspan=3, pady=10)

        ttk.Button(frame, text="Literalist", command=lambda: self.set_result('1')).grid(row=1, column=0, padx=5, pady=10)
        ttk.Button(frame, text="Contrarian", command=lambda: self.set_result('2')).grid(row=1, column=1, padx=5, pady=10)
        ttk.Button(frame, text="Comedian", command=lambda: self.set_result('3')).grid(row=1, column=2, padx=5, pady=10)

        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def set_result(self, value):
        self.result = value
        self.destroy()

    def cancel(self):
        self.result = '3'  # Default to Literalist if dialog is closed
        self.destroy()
