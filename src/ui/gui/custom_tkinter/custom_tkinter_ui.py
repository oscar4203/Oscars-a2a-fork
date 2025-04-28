class CustomTkinterUI(GameInterface):
    """CustomTkinter-based UI implementation for Apples to Apples."""

    def __init__(self, theme: str = "dark-blue", state_manager: Optional["GameStateManager"] = None):
        """Initialize the CustomTkinter UI."""
        # Set up CustomTkinter appearance
        ctk.set_appearance_mode("dark")  # Options: "light", "dark"
        ctk.set_default_color_theme(theme)  # Options: "blue", "dark-blue", "green"

        # Create the main window
        self.root = ctk.CTk()
        self.root.title("Apples to Apples")
        self.root.geometry("1200x800")

        # Default values
        self.points_to_win = 0
        self.state_manager = state_manager

        # Card selection tracking
        self.selected_card_index = None
        self.active_cards = []
        self.on_card_confirm = None
        self._selection_event = threading.Event()

        # Set up UI elements
        self.setup_ui()

        # Create handlers
        self._input_handler = CustomTkinterInputHandler(self.root)
        self._output_handler = CustomTkinterOutputHandler(self.root)
        self._output_handler.set_ui(self)

        # Player widgets dictionary
        self.player_widgets = {}

        logging.info(f"CustomTkinterUI initialized with theme: {theme}")

    @property
    def input_handler(self) -> InputHandler:
        """Return the input handler for this interface."""
        return self._input_handler

    @property
    def output_handler(self) -> OutputHandler:
        """Return the output handler for this interface."""
        return self._output_handler

    def setup_ui(self):
        """Set up the main UI structure."""
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header section
        self.header_frame = ctk.CTkFrame(self.main_frame)
        self.header_frame.pack(fill="x", pady=(0, 10))

        self.title_label = ctk.CTkLabel(self.header_frame, text="Apples to Apples",
                                       font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=5)

        # Game info section
        self.info_frame = ctk.CTkFrame(self.header_frame)
        self.info_frame.pack(fill="x", pady=5)

        self.game_label = ctk.CTkLabel(self.info_frame, text="Game: 0/0")
        self.game_label.pack(side="left", padx=10)

        self.round_label = ctk.CTkLabel(self.info_frame, text="Round: 0")
        self.round_label.pack(side="left", padx=10)

        self.points_label = ctk.CTkLabel(self.info_frame, text="Points to Win: 0")
        self.points_label.pack(side="left", padx=10)

        # Status message
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", pady=(0, 10))

        self.status_label = ctk.CTkLabel(self.status_frame, text="Welcome to Apples to Apples!",
                                        wraplength=800)
        self.status_label.pack(pady=5)

        # Main content area
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, pady=5)

        # Left side - Player info
        self.left_frame = ctk.CTkFrame(self.content_frame, width=150)
        self.left_frame.pack(side="left", fill="y", padx=(0, 10))

        self.left_players_frame = ctk.CTkFrame(self.left_frame)
        self.left_players_frame.pack(fill="y", expand=True, pady=5)

        # Center area
        self.center_frame = ctk.CTkFrame(self.content_frame)
        self.center_frame.pack(side="left", fill="both", expand=True)

        # Green apple section
        self.green_apple_frame = ctk.CTkFrame(self.center_frame)
        self.green_apple_frame.pack(fill="x", pady=(0, 10))

        self.green_apple_label = ctk.CTkLabel(self.green_apple_frame, text="Green Apple",
                                             font=ctk.CTkFont(weight="bold"))
        self.green_apple_label.pack(pady=2)

        self.green_apple_card_frame = ctk.CTkFrame(self.green_apple_frame)
        self.green_apple_card_frame.pack(pady=5)

        # Submitted cards section
        self.submitted_cards_frame = ctk.CTkFrame(self.center_frame)
        self.submitted_cards_frame.pack(fill="both", expand=True, pady=5)

        self.submitted_cards_label = ctk.CTkLabel(self.submitted_cards_frame, text="Submitted Red Apples",
                                                font=ctk.CTkFont(weight="bold"))
        self.submitted_cards_label.pack(pady=2)

        self.submitted_cards_container = ctk.CTkFrame(self.submitted_cards_frame)
        self.submitted_cards_container.pack(fill="both", expand=True, pady=5)

        # Right side - Player info
        self.right_frame = ctk.CTkFrame(self.content_frame, width=150)
        self.right_frame.pack(side="right", fill="y", padx=(10, 0))

        self.right_players_frame = ctk.CTkFrame(self.right_frame)
        self.right_players_frame.pack(fill="y", expand=True, pady=5)

        # Bottom area - Player's hand
        self.player_cards_frame = ctk.CTkFrame(self.main_frame)
        self.player_cards_frame.pack(fill="x", pady=(10, 0))

        self.player_cards_label = ctk.CTkLabel(self.player_cards_frame, text="Your Red Apples",
                                             font=ctk.CTkFont(weight="bold"))
        self.player_cards_label.pack(pady=2)

        self.player_cards_container = ctk.CTkFrame(self.player_cards_frame)
        self.player_cards_container.pack(fill="x", pady=5)

    def _handle_card_selection(self, index: int) -> None:
        """Handle a card being selected by clicking."""
        # Deselect previously selected card
        if self.selected_card_index is not None and self.active_cards:
            if 0 <= self.selected_card_index < len(self.active_cards):
                prev_card = self.active_cards[self.selected_card_index]
                prev_card.deselect()
            else:
                logging.warning(f"Previous selected index {self.selected_card_index} out of bounds.")

        # Select new card
        self.selected_card_index = index
        if 0 <= index < len(self.active_cards):
            card = self.active_cards[index]
            card.select()

            # Call the confirmation callback
            if self.on_card_confirm is not None:
                self.on_card_confirm()
        else:
            logging.error(f"Selected index {index} out of bounds.")

    def update_players(self, players: List["Agent"]) -> None:
        """Update the display of players."""
        self.player_widgets = {}

        # Clear existing player widgets
        for frame in [self.left_players_frame, self.right_players_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        # Distribute players to left and right sides
        left_players = players[:len(players)//2]
        right_players = players[len(players)//2:]

        # Create player widgets
        for player in left_players:
            widget = PlayerInfoWidget(
                self.left_players_frame,
                player.get_name(),
                player.get_points(),
                player.get_judge_status()
            )
            widget.pack(pady=5, fill="x")
            self.player_widgets[player] = widget

        for player in right_players:
            widget = PlayerInfoWidget(
                self.right_players_frame,
                player.get_name(),
                player.get_points(),
                player.get_judge_status()
            )
            widget.pack(pady=5, fill="x")
            self.player_widgets[player] = widget

        self.root.update_idletasks()

    def prompt_human_agent_choose_red_apple(self, player: "Agent", red_apples: List["RedApple"],
                                          green_apple: "GreenApple") -> int:
        """Prompt a player to select a red apple via direct GUI click."""
        self.selected_card_index = None  # Reset selection
        self._selection_event.clear()  # Reset event flag

        # Display cards
        self.output_handler.display_player_red_apples(player)

        # Define callback
        def on_selection_complete():
            self._selection_event.set()  # Signal that selection is done

        self.on_card_confirm = on_selection_complete

        # Wait for user to click a card
        self.root.update()
        self._selection_event.wait()  # Block until selection is complete

        # Get the final selection
        index = self.selected_card_index

        # Default to first card if no selection
        if index is None:
            logging.warning("No card selected, defaulting to first card.")
            index = 0

        return index

    def run(self) -> None:
        """Run the Tkinter main loop."""
        logging.info("Starting CustomTkinter main loop.")
        self.root.mainloop()

    def teardown(self) -> None:
        """Clean up resources."""
        logging.info("Tearing down CustomTkinter UI.")
        if hasattr(self, 'root') and self.root:
            self.root.quit()
            self.root.destroy()

    # --- GameInterface implementation methods ---
    # These methods just delegate to the appropriate handler

    def display_new_game_message(self) -> None:
        self.output_handler.display_new_game_message()

    def display_game_header(self, game_number: int, total_games: int) -> None:
        self.output_handler.display_game_header(game_number, total_games)

    def display_initializing_decks(self) -> None:
        self.output_handler.display_initializing_decks()

    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        self.output_handler.display_deck_sizes(green_deck_size, red_deck_size)

    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        self.output_handler.display_deck_loaded(deck_name, count)

    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        self.output_handler.display_expansion_deck_loaded(deck_name, count)

    def display_initializing_players(self) -> None:
        self.output_handler.display_initializing_players()

    def display_player_count(self, count: int) -> None:
        self.output_handler.display_player_count(count)

    def display_starting_judge(self, judge_name: str) -> None:
        self.output_handler.display_starting_judge(judge_name)

    def display_next_judge(self, judge_name: str) -> None:
        self.output_handler.display_next_judge(judge_name)

    def display_round_header(self, round_number: int) -> None:
        self.output_handler.display_round_header(round_number)

    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        self.output_handler.display_player_points(player_points)

    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        self.output_handler.display_green_apple(judge, green_apple)

    def display_player_red_apples(self, player: "Agent") -> None:
        self.output_handler.display_player_red_apples(player)

    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        self.output_handler.display_red_apple_chosen(player, red_apple)

    def display_winning_red_apple(self, judge: "Agent", red_apple: "RedApple") -> None:
        self.output_handler.display_winning_red_apple(judge, red_apple)

    def display_round_winner(self, winner: "Agent") -> None:
        self.output_handler.display_round_winner(winner)

    def display_game_winner(self, winner: "Agent") -> None:
        self.output_handler.display_game_winner(winner)

    def display_game_time(self, minutes: int, seconds: int) -> None:
        self.output_handler.display_game_time(minutes, seconds)

    def display_resetting_models(self) -> None:
        self.output_handler.display_resetting_models()

    def display_training_green_apple(self, green_apple: "GreenApple") -> None:
        self.output_handler.display_training_green_apple(green_apple)

    def prompt_keep_players_between_games(self) -> bool:
        return self.input_handler.prompt_yes_no("Do you want to keep the same players as last game?")

    def prompt_starting_judge(self, player_count: int) -> int:
        return self.input_handler.prompt_starting_judge(player_count)

    def prompt_player_type(self, player_number: int) -> str:
        return self.input_handler.prompt_player_type(player_number)

    def prompt_human_player_name(self) -> str:
        return self.input_handler.prompt_human_player_name()

    def prompt_ai_model_type(self) -> str:
        return self.input_handler.prompt_ai_model_type()

    def prompt_ai_archetype(self) -> str:
        return self.input_handler.prompt_ai_archetype()

    def prompt_training_model_type(self) -> str:
        return self.input_handler.prompt_training_model_type()

    def prompt_training_pretrained_type(self) -> str:
        return self.input_handler.prompt_training_pretrained_type()

    def prompt_judge_draw_green_apple(self, judge: "Agent") -> None:
        self.output_handler.prompt_judge_draw_green_apple(judge)

    def prompt_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        self.output_handler.prompt_select_red_apple(player, green_apple)

    def prompt_training_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        self.output_handler.prompt_training_select_red_apple(player, green_apple)

    def prompt_training_select_bad_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        self.output_handler.prompt_training_select_bad_red_apple(player, green_apple)

    def prompt_judge_select_winner(self, judge: "Agent") -> None:
        self.output_handler.prompt_judge_select_winner(judge)
