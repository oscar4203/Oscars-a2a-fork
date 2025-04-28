class AppleCard(ctk.CTkFrame):
    """Base class for Apple cards."""
    def __init__(self, master, bg_color: str, text: str, description: str = None,
                width: int = 120, height: int = 180, command: Optional[Callable] = None,
                highlight: bool = False):
        super().__init__(master, width=width, height=height, fg_color=bg_color,
                         border_width=2, border_color="gold" if highlight else "gray")

        self.bg_color = bg_color
        self.command = command
        self.highlight = highlight

        # Make sure the card doesn't shrink
        self.grid_propagate(False)

        # Create the text label
        self.text_label = ctk.CTkLabel(
            self,
            text=text,
            wraplength=100,
            text_color=("white", "white"),
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.text_label.place(relx=0.5, rely=0.3, anchor="center")

        # Add description if provided
        if description:
            self.desc_label = ctk.CTkLabel(
                self,
                text=description,
                wraplength=100,
                text_color=("white", "white"),
                font=ctk.CTkFont(size=10)
            )
            self.desc_label.place(relx=0.5, rely=0.7, anchor="center")

        # Make card clickable if command is provided
        if command:
            self.bind("<Button-1>", lambda event: command())

    def select(self):
        """Highlight this card as selected."""
        self.configure(border_color="blue", border_width=3)

    def deselect(self):
        """Remove highlight from this card."""
        self.configure(border_color="gray" if not self.highlight else "gold", border_width=2)


class GreenAppleCard(AppleCard):
    """Widget for Green Apple cards."""
    def __init__(self, master, adjective: str, synonyms: str = None, command: Optional[Callable] = None,
                highlight: bool = False):
        super().__init__(
            master,
            "#4CAF50",  # Green color
            adjective,
            synonyms,
            command=command,
            highlight=highlight
        )


class RedAppleCard(AppleCard):
    """Widget for Red Apple cards."""
    def __init__(self, master, noun: str, description: str = None, command: Optional[Callable] = None,
                highlight: bool = False):
        super().__init__(
            master,
            "#F44336",  # Red color
            noun,
            description,
            command=command,
            highlight=highlight
        )


class PlayerInfoWidget(ctk.CTkFrame):
    """Widget for displaying player information."""
    def __init__(self, master, player_name: str, points: int = 0, is_judge: bool = False,
                is_current_player: bool = False):
        # Choose the appropriate background color
        bg_color = "#2B2B2B"  # Default dark background
        if is_judge:
            bg_color = "#5D4037"  # Brown for judge
        elif is_current_player:
            bg_color = "#1565C0"  # Blue for current player

        super().__init__(master, fg_color=bg_color, corner_radius=10)

        self.is_judge = is_judge

        # Player name
        self.name_label = ctk.CTkLabel(
            self,
            text=player_name,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.name_label.pack(pady=(10, 5))

        # Points display
        self.points_label = ctk.CTkLabel(
            self,
            text=f"Points: {points}"
        )
        self.points_label.pack(pady=5)

        # Judge indicator if applicable
        if is_judge:
            self.judge_label = ctk.CTkLabel(
                self,
                text="JUDGE",
                text_color="#FFD700",  # Gold color
                font=ctk.CTkFont(size=12, weight="bold")
            )
            self.judge_label.pack(pady=(5, 10))

    def update_points(self, points: int) -> None:
        """Update the player's points."""
        self.points_label.configure(text=f"Points: {points}")

    def set_as_judge(self, is_judge: bool = True) -> None:
        """Mark or unmark the player as judge."""
        self.is_judge = is_judge
        if is_judge:
            self.configure(fg_color="#5D4037")  # Brown for judge
            if not hasattr(self, 'judge_label'):
                self.judge_label = ctk.CTkLabel(
                    self,
                    text="JUDGE",
                    text_color="#FFD700",  # Gold color
                    font=ctk.CTkFont(size=12, weight="bold")
                )
                self.judge_label.pack(pady=(5, 10))
        else:
            self.configure(fg_color="#2B2B2B")  # Default dark background
            if hasattr(self, 'judge_label'):
                self.judge_label.pack_forget()

    def set_as_current_player(self, is_current: bool = True) -> None:
        """Mark or unmark the player as current player."""
        if is_current:
            self.configure(fg_color="#1565C0")  # Blue for current player
        else:
            if self.is_judge:
                self.configure(fg_color="#5D4037")  # Brown for judge
            else:
                self.configure(fg_color="#2B2B2B")  # Default dark background
