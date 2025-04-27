# Description: Custom widgets for Tkinter UI in Apples to Apples.

# Standard Libraries
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, TYPE_CHECKING

# Type Checking
if TYPE_CHECKING:
    from src.apples.apples import GreenApple, RedApple


class AppleCard(tk.Frame):
    """Base widget representing an Apple card."""
    def __init__(self, master, bg_color: str, width: int = 120, height: int = 180, command: Optional[Callable] = None):
        super().__init__(master, bg=bg_color, width=width, height=height, bd=2, relief=tk.RAISED)

        self.command = command
        self.bg_color = bg_color

        # Make card clickable if command provided
        if command:
            self.bind("<Button-1>", lambda event: command())

        # Prevent card from shrinking
        self.pack_propagate(False)

    def _create_label(self, text: str, wraplength: int = 100):
        """Create and place the text label on the card."""
        self.label = tk.Label(self,
                            text=text,
                            bg=self.bg_color,
                            fg="white",
                            wraplength=wraplength,
                            justify=tk.CENTER)
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


class GreenAppleCard(AppleCard):
    """Widget representing a Green Apple card (adjective)."""
    def __init__(self, master, adjective: str, synonyms: Optional[str] = None, command: Optional[Callable] = None):
        bg_color = "#4CAF50"  # Green
        super().__init__(master, bg_color, command=command)

        self.adjective = adjective
        self.synonyms = synonyms

        # Format and display text
        display_text = f"{adjective}"
        if synonyms:
            display_text += f"\n\n{synonyms}"

        self._create_label(display_text)

    @classmethod
    def from_green_apple(cls, master, green_apple: "GreenApple", command: Optional[Callable] = None):
        """Create a GreenAppleCard from a GreenApple object."""
        synonyms_list = green_apple.get_synonyms()
        synonyms = ", ".join(synonyms_list) if synonyms_list is not None else None
        return cls(master, green_apple.get_adjective(), synonyms, command)


class RedAppleCard(AppleCard):
    """Widget representing a Red Apple card (noun)."""
    def __init__(self, master, noun: str, description: Optional[str] = None, command: Optional[Callable] = None):
        bg_color = "#F44336"  # Red
        super().__init__(master, bg_color, command=command)

        self.noun = noun
        self.description = description

        # Format and display text
        display_text = f"{noun}"
        if description:
            display_text += f"\n\n{description}"

        self._create_label(display_text)

    @classmethod
    def from_red_apple(cls, master, red_apple: "RedApple", command: Optional[Callable] = None):
        """Create a RedAppleCard from a RedApple object."""
        return cls(master, red_apple.get_noun(), red_apple.get_description(), command)


class PlayerInfoWidget(tk.Frame):
    """Widget displaying player information."""
    def __init__(self, master, player_name: str, points: int = 0, is_judge: bool = False,
                 is_current_player: bool = False, width: int = 140, height: int = 80):
        bg_color = "#E0E0E0"  # Default light gray
        if is_judge:
            bg_color = "#FFD700"  # Gold for judge
        elif is_current_player:
            bg_color = "#90CAF9"  # Light blue for current player

        super().__init__(master, bg=bg_color, bd=1, relief=tk.RAISED, padx=5, pady=5, width=width, height=height)

        self.is_judge = is_judge

        # Player name
        self.name_label = tk.Label(self, text=player_name, bg=bg_color, font=("Arial", 10, "bold"))
        self.name_label.pack(pady=(0, 2))

        # Points display
        self.points_label = tk.Label(self, text=f"Points: {points}", bg=bg_color)
        self.points_label.pack()

        # Judge label if applicable
        if is_judge:
            self.judge_label = tk.Label(self, text="JUDGE", bg=bg_color, fg="red", font=("Arial", 8, "bold"))
            self.judge_label.pack()

        # Prevent widget from shrinking
        self.pack_propagate(False)

    def update_points(self, points: int) -> None:
        """Update the player's points."""
        self.points_label.config(text=f"Points: {points}")

    def set_as_judge(self, is_judge: bool = True) -> None:
        """Mark or unmark the player as judge."""
        self.is_judge = is_judge
        if is_judge:
            self.config(bg="#FFD700")
            self.name_label.config(bg="#FFD700")
            self.points_label.config(bg="#FFD700")
            if not hasattr(self, 'judge_label'):
                self.judge_label = tk.Label(self, text="JUDGE", bg="#FFD700", fg="red", font=("Arial", 8, "bold"))
                self.judge_label.pack()
        else:
            self.config(bg="#E0E0E0")
            self.name_label.config(bg="#E0E0E0")
            self.points_label.config(bg="#E0E0E0")
            if hasattr(self, 'judge_label'):
                self.judge_label.pack_forget()

    def set_as_current_player(self, is_current: bool = True) -> None:
        """Mark or unmark the player as current player."""
        if is_current:
            self.config(bg="#90CAF9")
            self.name_label.config(bg="#90CAF9")
            self.points_label.config(bg="#90CAF9")
            if hasattr(self, 'judge_label'):
                self.judge_label.config(bg="#90CAF9")
        else:
            if self.is_judge:
                self.config(bg="#FFD700")
                self.name_label.config(bg="#FFD700")
                self.points_label.config(bg="#FFD700")
                if hasattr(self, 'judge_label'):
                    self.judge_label.config(bg="#FFD700")
            else:
                self.config(bg="#E0E0E0")
                self.name_label.config(bg="#E0E0E0")
                self.points_label.config(bg="#E0E0E0")
                if hasattr(self, 'judge_label'):
                    self.judge_label.config(bg="#E0E0E0")
