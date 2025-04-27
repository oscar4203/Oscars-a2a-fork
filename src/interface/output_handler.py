# Description: Abstract output handling for Apples to Apples game. Terminal-based and Tkinter-based implementations.

# Standard Libraries
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING
import logging

# Third-party Libraries
import tkinter as tk
from tkinter import messagebox

# Local Modules
from src.ui.gui.tkinter_widgets import GreenAppleCard, RedAppleCard

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    from src.ui.gui.tkinter_ui import TkinterUI


class OutputHandler(ABC):
    """Abstract class for handling output in various UIs."""

    @abstractmethod
    def display_message(self, message: str) -> None:
        """
        Display a general message to the user.

        Args:
            message: The message to display
        """
        pass

    @abstractmethod
    def display_error(self, message: str) -> None:
        """
        Display an error message to the user.

        Args:
            message: The error message to display
        """
        pass

    @abstractmethod
    def display_new_game_message(self) -> None:
        """Display a message indicating a new game is starting."""
        pass

    @abstractmethod
    def display_game_header(self, game_number: int, total_games: int) -> None:
        """
        Display the game header with game number information.

        Args:
            game_number: The current game number
            total_games: The total number of games to play
        """
        pass

    @abstractmethod
    def display_round_header(self, round_number: int) -> None:
        """
        Display the round header with round number information.

        Args:
            round_number: The current round number
        """
        pass

    @abstractmethod
    def display_initializing_decks(self) -> None:
        """Display message about initializing decks."""
        pass

    @abstractmethod
    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        """
        Display message about a deck being loaded.

        Args:
            deck_name: The name of the deck
            count: The number of cards loaded
        """
        pass

    @abstractmethod
    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        """
        Display message about an expansion deck being loaded.

        Args:
            deck_name: The name of the expansion deck
            count: The number of expansion cards loaded
        """
        pass

    @abstractmethod
    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        """
        Display the sizes of the green and red apple decks.

        Args:
            green_deck_size: The size of the green apple deck
            red_deck_size: The size of the red apple deck
        """
        pass

    @abstractmethod
    def display_initializing_players(self) -> None:
        """Display message about initializing players."""
        pass

    @abstractmethod
    def display_player_count(self, count: int) -> None:
        """
        Display the number of players in the game.

        Args:
            count: The number of players
        """
        pass

    @abstractmethod
    def display_starting_judge(self, judge_name: str) -> None:
        """
        Display the name of the starting judge.

        Args:
            judge_name: The name of the starting judge
        """
        pass

    @abstractmethod
    def display_next_judge(self, judge_name: str) -> None:
        """
        Display the name of the next judge.

        Args:
            judge_name: The name of the next judge
        """
        pass

    @abstractmethod
    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        """
        Display all players' points.

        Args:
            player_points: List of tuples containing player names and their point totals
        """
        pass

    @abstractmethod
    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        """
        Display the green apple card in play.

        Args:
            judge: The judge who drew the green apple
            green_apple: The green apple card to display
        """
        pass

    @abstractmethod
    def display_player_red_apples(self, player: "Agent") -> None:
        """
        Display the red apples held by a player.

        Args:
            player: The player whose red apples to display
        """
        pass

    @abstractmethod
    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """
        Display that a player has chosen a red apple.

        Args:
            player: The player who chose the red apple
            red_apple: The chosen red apple card
        """
        pass

    @abstractmethod
    def display_winning_red_apple(self, judge_name: str, red_apple: "RedApple") -> None:
        """
        Display the winning red apple card.

        Args:
            judge_name: The name of the judge who made the selection
            red_apple: The winning red apple card
        """
        pass

    @abstractmethod
    def display_round_winner(self, winner_name: str) -> None:
        """
        Display the round winner.

        Args:
            winner_name: The name of the round winner
        """
        pass

    @abstractmethod
    def display_game_winner(self, winner_name: str) -> None:
        """
        Display the game winner.

        Args:
            winner_name: The name of the game winner
        """
        pass

    @abstractmethod
    def display_game_time(self, minutes: int, seconds: int) -> None:
        """
        Display the total game time.

        Args:
            minutes: Minutes elapsed
            seconds: Seconds elapsed
        """
        pass

    @abstractmethod
    def display_resetting_models(self) -> None:
        """Display message about resetting AI opponent models."""
        pass

    @abstractmethod
    def display_training_green_apple(self, adjective: str) -> None:
        """
        Display the green apple card in training mode.

        Args:
            adjective: The adjective of the green apple
        """
        pass

    @abstractmethod
    def prompt_judge_draw_green_apple(self, judge_name: str) -> None:
        """
        Display prompt for the judge to draw a green apple.

        Args:
            judge_name: The name of the judge
        """
        pass

    @abstractmethod
    def prompt_select_red_apple(self, player_name: str) -> None:
        """
        Display prompt for a player to select a red apple.

        Args:
            player_name: The name of the player
        """
        pass

    @abstractmethod
    def prompt_training_select_red_apple(self, player: "Agent") -> None:
        """
        Display prompt for a player to select a good red apple in training mode.

        Args:
            player: The player selecting the card
        """
        pass

    @abstractmethod
    def prompt_training_select_bad_red_apple(self, player_name: str) -> None:
        """
        Display prompt for a player to select a bad red apple in training mode.

        Args:
            player_name: The name of the player
        """
        pass

    @abstractmethod
    def prompt_judge_select_winner(self, judge_name: str) -> None:
        """
        Display prompt for the judge to select the winning red apple.

        Args:
            judge_name: The name of the judge
        """
        pass

    @abstractmethod
    def display_submitted_red_apples(self, red_apples: List[Tuple["Agent", "RedApple"]]) -> None:
        """
        Display all submitted red apple cards.

        Args:
            red_apples: List of tuples with agents and their submitted red apples
        """
        pass

    @abstractmethod
    def display_agent_drew_cards(self, agent_name: str, count: int) -> None:
        """
        Display message that an agent drew cards.

        Args:
            agent_name: Name of the agent
            count: Number of cards drawn
        """
        pass

    @abstractmethod
    def display_agent_cant_draw_cards(self, agent_name: str) -> None:
        """
        Display message that an agent cannot draw more cards.

        Args:
            agent_name: Name of the agent
        """
        pass

    @abstractmethod
    def display_model_reset(self, agent_name: str, opponent_name: str) -> None:
        """
        Display message that an agent's model for an opponent was reset.

        Args:
            agent_name: Name of the agent
            opponent_name: Name of the opponent
        """
        pass

    @abstractmethod
    def log_agent_chose_red_apple(self, agent_name: str, red_apple: "RedApple") -> None:
        """
        Log that an agent chose a red apple.

        Args:
            agent_name: Name of the agent
            red_apple: The chosen red apple
        """
        pass

    @abstractmethod
    def log_model_training(self, agent_name: str, opponent_name: str,
                          green_apple: "GreenApple", winning_red_apple: "RedApple",
                          losing_red_apples: List["RedApple"]) -> None:
        """
        Log model training information.

        Args:
            agent_name: Name of the agent being trained
            opponent_name: Name of the opponent/judge
            green_apple: The green apple in play
            winning_red_apple: The winning red apple
            losing_red_apples: List of losing red apples
        """
        pass

    @abstractmethod
    def log_error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: Error message to log
        """
        pass

    @abstractmethod
    def log_debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message: Debug message to log
        """
        pass


class TerminalOutputHandler(OutputHandler):
    """Implementation of OutputHandler for terminal/console interface."""

    def __init__(self, print_in_terminal: bool = True):
        """
        Initialize the terminal output handler.

        Args:
            print_in_terminal: Whether to print output to the terminal
        """
        self.print_in_terminal = print_in_terminal

    def display_message(self, message: str) -> None:
        """Display a general message via terminal output."""
        if self.print_in_terminal:
            print(message)

    def display_error(self, message: str) -> None:
        """Display an error message via terminal output."""
        if self.print_in_terminal:
            print(f"ERROR: {message}")

    def display_new_game_message(self) -> None:
        """Display a message indicating a new game is starting via terminal output."""
        if self.print_in_terminal:
            message = "\nStarting new 'Apples to Apples' game."
            print(message)
            logging.info(message)

    def display_game_header(self, game_number: int, total_games: int) -> None:
        """Display the game header via terminal output."""
        if self.print_in_terminal:
            message = f"\n------------- GAME {game_number} of {total_games} -------------"
            print(message)
            logging.info(message)

    def display_round_header(self, round_number: int) -> None:
        """Display the round header via terminal output."""
        if self.print_in_terminal:
            message = f"\n===================\nROUND {round_number}:\n===================\n"
            print(message)
            logging.info(message)

    def display_initializing_decks(self) -> None:
        """Display message about initializing decks via terminal output."""
        if self.print_in_terminal:
            print("Initializing decks.")

    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about a deck being loaded via terminal output."""
        if self.print_in_terminal:
            print(f"Loaded {count} {deck_name.lower()}.")

    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about an expansion deck being loaded via terminal output."""
        if self.print_in_terminal:
            print(f"Loaded {count} {deck_name.lower()} from the expansion.")

    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        """Display the sizes of the decks via terminal output."""
        if self.print_in_terminal:
            print(f"Size of green apples deck: {green_deck_size}")
            print(f"Size of red apples deck: {red_deck_size}")

    def display_initializing_players(self) -> None:
        """Display message about initializing players via terminal output."""
        if self.print_in_terminal:
            message = "Initializing players."
            print(message)
            logging.info(message)

    def display_player_count(self, count: int) -> None:
        """Display the number of players via terminal output."""
        if self.print_in_terminal:
            message = f"\nThere are {count} player{'s' if count != 1 else ''} in the game."
            print(message)
            logging.info(message)

    def display_starting_judge(self, judge_name: str) -> None:
        """Display the name of the starting judge via terminal output."""
        if self.print_in_terminal:
            print(f"\n{judge_name} is the starting judge.")

    def display_next_judge(self, judge_name: str) -> None:
        """Display the name of the next judge via terminal output."""
        if self.print_in_terminal:
            print(f"{judge_name} is the next judge.")

    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        """Display all players' points via terminal output."""
        if self.print_in_terminal:
            for name, points in player_points:
                print(f"{name}: {points} points")

    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        """Display the green apple card via terminal output."""
        message = f"\n{judge.get_name()} drew the green apple '{green_apple}'."
        logging.info(message)
        if self.print_in_terminal:
            print(message)

    def display_player_red_apples(self, player: "Agent") -> None:
        """Display the red apples held by a player via terminal output."""
        if self.print_in_terminal:
            message1 = f"\n{player.get_name()}'s red apples:"
            print(message1)
            logging.info(message1)
            for i, red_apple in enumerate(player.get_red_apples()):
                message2 = f"{i + 1}. {red_apple}"
                print(message2)
                logging.info(message2)

    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """Display the red apple chosen by a player via terminal output."""
        if self.print_in_terminal:
            message = f"{player.get_name()} chose the red apple '{red_apple.get_noun()}'."
            print(message)
            logging.info(message)

    def display_winning_red_apple(self, judge: "Agent", red_apple: "RedApple") -> None:
        """Display the winning red apple card via terminal output."""
        if self.print_in_terminal:
            message = f"{judge.get_name()} chose the winning red apple '{red_apple.get_noun()}'."
            print(message)
            logging.info(message)

    def display_round_winner(self, winner: "Agent") -> None:
        """Display the round winner via terminal output."""
        if self.print_in_terminal:
            message = f"\n***{winner.get_name()} has won the round!***"
            print(message)
            logging.info(message)

    def display_game_winner(self, winner: "Agent") -> None:
        """Display the game winner via terminal output."""
        if self.print_in_terminal:
            message = f"### {winner.get_name()} has won the game! ###"
            print(message)
            logging.info(message)

    def display_game_time(self, minutes: int, seconds: int) -> None:
        """Display the total game time via terminal output."""
        if self.print_in_terminal:
            message = f"Total game time: {minutes} minute(s), {seconds} second(s)"
            print(message)
            logging.info(message)

    def display_resetting_models(self) -> None:
        """Display message about resetting AI opponent models via terminal output."""
        if self.print_in_terminal:
            print("Resetting opponent models for all AI agents.")

    def display_training_green_apple(self, green_apple: "GreenApple") -> None:
        """Display the green apple card in training mode via terminal output."""
        if self.print_in_terminal:
            message = f"\nThe green apple is '{green_apple.get_adjective()}'."
            print(message)
            logging.info(message)

    # === Prompts for Player Actions ===

    def prompt_judge_draw_green_apple(self, judge: "Agent") -> None:
        """Display prompt for the judge to draw a green apple via terminal output."""
        if self.print_in_terminal:
            message = f"\n{judge.get_name()}, please draw a green apple."
            print(message)
            logging.info(message)

    def prompt_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Display prompt for a player to select a red apple via terminal output."""
        if self.print_in_terminal:
            message = f"\n{player.get_name()}, please select a red apple."
            print(message)
            logging.info(message)

    def prompt_training_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Display prompt for a player to select a good red apple in training mode via terminal output."""
        if self.print_in_terminal:
            message = f"\n{player.get_name()}, please select a good red apple."
            print(message)
            logging.info(message)

    def prompt_training_select_bad_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Display prompt for a player to select a bad red apple in training mode via terminal output."""
        if self.print_in_terminal:
            message = f"\n{player.get_name()}, please select a bad red apple."
            print(message)
            logging.info(message)

    def prompt_judge_select_winner(self, judge: "Agent") -> None:
        """Display prompt for the judge to select the winning red apple via terminal output."""
        if self.print_in_terminal:
            message = f"\n{judge.get_name()}, please select the winning red apple."
            print(message)
            logging.info(message)

    def display_submitted_red_apples(self, red_apples: List[Tuple["Agent", "RedApple"]]) -> None:
        """Display all submitted red apple cards via terminal output."""
        if self.print_in_terminal:
            print("Red cards submitted by the other agents:")
            for i, (agent, red_apple) in enumerate(red_apples):
                print(f"{i + 1}. {red_apple}")

    def display_agent_drew_cards(self, agent_name: str, count: int) -> None:
        """Display message that an agent drew cards via terminal output."""
        if self.print_in_terminal:
            message = f"{agent_name} picked up {count} red apple{'s' if count != 1 else ''}."
            print(message)

    def display_agent_cant_draw_cards(self, agent_name: str) -> None:
        """Display message that an agent cannot draw more cards via terminal output."""
        if self.print_in_terminal:
            message = f"{agent_name} cannot pick up any more red apples. Agent already has enough red apples."
            print(message)

    def display_model_reset(self, agent_name: str, opponent_name: str) -> None:
        """Display message that an agent's model for an opponent was reset via terminal output."""
        if self.print_in_terminal:
            message = f"Reset {opponent_name}'s model."
            print(message)

    def log_agent_chose_red_apple(self, agent_name: str, red_apple: "RedApple") -> None:
        """Log that an agent chose a red apple."""
        logging.info(f"{agent_name} chose the red apple '{red_apple}'.")

    def log_model_training(self, agent_name: str, opponent_name: str,
                          green_apple: "GreenApple", winning_red_apple: "RedApple",
                          losing_red_apples: List["RedApple"]) -> None:
        """Log model training information."""
        message = f"Trained {agent_name}'s model for '{opponent_name}'. Green apple '{green_apple}'. Winning red apple '{winning_red_apple}'."
        if losing_red_apples:
            message += f" Losing red apples: {losing_red_apples}."
        logging.debug(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        logging.error(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        logging.debug(message)


class TkinterOutputHandler(OutputHandler):
    """Implementation of OutputHandler for Tkinter GUI interface."""

    def __init__(self, root: tk.Tk, ui: "TkinterUI"):
        """
        Initialize the Tkinter output handler.

        Args:
            root: The Tkinter root window
            ui: The TkinterUI instance
        """
        self.root = root
        self.ui = ui
        # List to track submitted red apples for UI display purposes
        self._submitted_red_apples: List[Tuple["Agent", "RedApple"]] = []

    def display_message(self, message: str) -> None:
        """Display a general message in the UI."""
        self.ui.status_label.config(text=message)
        logging.info(message)
        self.root.update_idletasks()

    def display_error(self, message: str) -> None:
        """Display an error message in the UI."""
        self.ui.status_label.config(text=f"ERROR: {message}")
        messagebox.showerror("Error", message)
        logging.error(message)
        self.root.update_idletasks()

    def display_new_game_message(self) -> None:
        """Display a message indicating a new game is starting."""
        message = "Starting new 'Apples to Apples' game."
        self.display_message(message)

    def display_game_header(self, game_number: int, total_games: int) -> None:
        """Display the game header with game number information."""
        # Update the UI label without needing to store state in the UI
        self.ui.game_label.config(text=f"Game: {game_number}/{total_games}")
        self.root.update_idletasks()

    def display_round_header(self, round_number: int) -> None:
        """Display the round header with round number information."""
        # Update the UI label without needing to store state in the UI
        self.ui.round_label.config(text=f"Round: {round_number}")
        self.root.update_idletasks()

    def display_initializing_decks(self) -> None:
        """Display message about initializing decks."""
        self.display_message("Initializing decks...")

    def display_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about a deck being loaded."""
        self.display_message(f"Loaded {count} {deck_name.lower()}.")

    def display_expansion_deck_loaded(self, deck_name: str, count: int) -> None:
        """Display message about an expansion deck being loaded."""
        self.display_message(f"Loaded {count} {deck_name.lower()} from the expansion.")

    def display_deck_sizes(self, green_deck_size: int, red_deck_size: int) -> None:
        """Display the sizes of the green and red apple decks."""
        self.display_message(f"Green apples: {green_deck_size}, Red apples: {red_deck_size}")

    def display_initializing_players(self) -> None:
        """Display message about initializing players."""
        self.display_message("Initializing players...")

    def display_player_count(self, count: int) -> None:
        """Display the number of players in the game."""
        self.display_message(f"There are {count} players in the game.")
        # Points to win equals the number of players
        self.ui.points_label.config(text=f"Points to Win: {count}")
        self.root.update_idletasks()

    def display_starting_judge(self, judge_name: str) -> None:
        """Display the name of the starting judge."""
        self.display_message(f"{judge_name} is the starting judge.")

        # Update player widgets to show the judge
        for player, widget in self.ui.player_widgets.items():
            widget.set_as_judge(player.get_name() == judge_name)

        self.root.update_idletasks()

    def display_next_judge(self, judge_name: str) -> None:
        """Display the name of the next judge."""
        self.display_message(f"{judge_name} is the next judge.")

        # Update player widgets to show the judge
        for player, widget in self.ui.player_widgets.items():
            widget.set_as_judge(player.get_name() == judge_name)

        self.root.update_idletasks()

    def display_player_points(self, player_points: List[Tuple[str, int]]) -> None:
        """Display all players' points."""
        for name, points in player_points:
            for player, widget in self.ui.player_widgets.items():
                if player.get_name() == name:
                    widget.update_points(points)

        self.root.update_idletasks()

    def display_green_apple(self, judge: "Agent", green_apple: "GreenApple") -> None:
        """Display the green apple card in play."""
        # Clear existing card
        for widget in self.ui.green_apple_card_frame.winfo_children():
            widget.destroy()

        # Create new green apple card
        card = GreenAppleCard.from_green_apple(
            self.ui.green_apple_card_frame,
            green_apple
        )
        card.pack()

        message = f"{judge.get_name()} drew the green apple '{green_apple.get_adjective()}'."
        self.display_message(message)
        self.root.update_idletasks()

    def display_player_red_apples(self, player: "Agent") -> None:
        """Display the red apples held by a player."""
        # Update player widgets to show the current player
        for p, widget in self.ui.player_widgets.items():
            widget.set_as_current_player(p == player)

        # Clear existing cards
        for widget in self.ui.player_cards_container.winfo_children():
            widget.destroy()

        # Create container for cards
        cards_frame = tk.Frame(self.ui.player_cards_container)
        cards_frame.pack(fill=tk.X)

        # Get red apples from player
        red_apples = player.get_red_apples()

        # Create card widgets
        for i, apple in enumerate(red_apples):
            card = RedAppleCard.from_red_apple(
                cards_frame,
                apple
            )
            card.grid(row=i//7, column=i%7, padx=5, pady=5)

        self.root.update_idletasks()

    def display_red_apple_chosen(self, player: "Agent", red_apple: "RedApple") -> None:
        """Display that a player has chosen a red apple."""
        message = f"{player.get_name()} chose the red apple '{red_apple.get_noun()}'."
        self.display_message(message)

        # Add to submitted cards (in our local tracking list)
        self._submitted_red_apples.append((player, red_apple))
        self.display_submitted_red_apples(self._submitted_red_apples)

        self.root.update_idletasks()

    def display_winning_red_apple(self, judge: "Agent", red_apple: "RedApple") -> None:
        """Display the winning red apple card."""
        message = f"{judge.get_name()} chose the winning red apple '{red_apple.get_noun()}'."
        self.display_message(message)

        # Highlight the winning card in the submitted cards area
        for widget in self.ui.submitted_cards_container.winfo_children():
            widget.destroy()

        # Display submitted cards with winner highlighted
        winner_found = False
        for player, apple in self._submitted_red_apples:
            is_winner = (apple == red_apple)
            frame = tk.Frame(self.ui.submitted_cards_container)
            frame.pack(side=tk.LEFT, padx=5, pady=5)

            if is_winner:
                winner_found = True
                # Create card with special styling for winner
                card = RedAppleCard.from_red_apple(
                    frame,
                    apple
                )
                card.config(highlightbackground="gold", highlightthickness=3)
                card.pack()

                # Add winner label
                winner_label = tk.Label(frame, text="WINNER", foreground="green", font=("Arial", 10, "bold"))
                winner_label.pack()
            else:
                # Regular card for non-winners
                card = RedAppleCard.from_red_apple(
                    frame,
                    apple
                )
                card.pack()

        # Clear submitted red apples after a winner is chosen
        self._submitted_red_apples = []

        self.root.update_idletasks()

    def display_round_winner(self, winner: "Agent") -> None:
        """Display the round winner."""
        message = f"{winner.get_name()} has won the round!"
        self.display_message(message)
        messagebox.showinfo("Round Winner", message)

    def display_game_winner(self, winner: "Agent") -> None:
        """Display the game winner."""
        message = f"{winner.get_name()} has won the game!"
        self.display_message(message)
        messagebox.showinfo("Game Winner", message)

    def display_game_time(self, minutes: int, seconds: int) -> None:
        """Display the total game time."""
        message = f"Total game time: {minutes} minute(s), {seconds} second(s)"
        self.display_message(message)

    def display_resetting_models(self) -> None:
        """Display message about resetting AI opponent models."""
        self.display_message("Resetting opponent models for all AI agents.")

    def display_training_green_apple(self, green_apple: "GreenApple") -> None:
        """Display the green apple card in training mode."""
        message = f"The green apple is '{green_apple.get_adjective()}'."
        self.display_message(message)

    def prompt_judge_draw_green_apple(self, judge: "Agent") -> None:
        """Display prompt for the judge to draw a green apple."""
        message = f"{judge.get_name()}, please draw a green apple."
        self.display_message(message)

    def prompt_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Display prompt for a player to select a red apple."""
        message = f"{player.get_name()}, please select a red apple."
        self.display_message(message)
        self.display_player_red_apples(player)

    def prompt_training_select_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Display prompt for a player to select a good red apple in training mode."""
        message = f"{player.get_name()}, please select a good red apple."
        self.display_message(message)
        self.display_player_red_apples(player)

    def prompt_training_select_bad_red_apple(self, player: "Agent", green_apple: "GreenApple") -> None:
        """Display prompt for a player to select a bad red apple in training mode."""
        message = f"{player.get_name()}, please select a bad red apple."
        self.display_message(message)
        self.display_player_red_apples(player)

    def prompt_judge_select_winner(self, judge: "Agent") -> None:
        """Display prompt for the judge to select the winning red apple."""
        message = f"{judge.get_name()}, please select the winning red apple."
        self.display_message(message)

    def display_submitted_red_apples(self, red_apples: List[Tuple["Agent", "RedApple"]]) -> None:
        """Display all submitted red apple cards."""
        # Clear existing cards
        for widget in self.ui.submitted_cards_container.winfo_children():
            widget.destroy()

        # Create card widgets for each submitted apple
        for player, apple in red_apples:
            frame = tk.Frame(self.ui.submitted_cards_container)
            frame.pack(side=tk.LEFT, padx=5, pady=5)

            card = RedAppleCard.from_red_apple(
                frame,
                apple
            )
            card.pack()

            # Add player name label
            player_label = tk.Label(frame, text=player.get_name())
            player_label.pack()

        self.root.update_idletasks()

    def display_agent_drew_cards(self, agent_name: str, count: int) -> None:
        """Display message that an agent drew cards."""
        message = f"{agent_name} picked up {count} red apple{'s' if count != 1 else ''}."
        self.display_message(message)

    def display_agent_cant_draw_cards(self, agent_name: str) -> None:
        """Display message that an agent cannot draw more cards."""
        message = f"{agent_name} cannot pick up any more red apples."
        self.display_message(message)

    def display_model_reset(self, agent_name: str, opponent_name: str) -> None:
        """Display message that an agent's model for an opponent was reset."""
        message = f"Reset {opponent_name}'s model."
        self.display_message(message)

    def log_agent_chose_red_apple(self, agent_name: str, red_apple: "RedApple") -> None:
        """Log that an agent chose a red apple."""
        logging.info(f"{agent_name} chose the red apple '{red_apple}'.")

    def log_model_training(self, agent_name: str, opponent_name: str,
                          green_apple: "GreenApple", winning_red_apple: "RedApple",
                          losing_red_apples: List["RedApple"]) -> None:
        """Log model training information."""
        message = f"Trained {agent_name}'s model for '{opponent_name}'. Green apple '{green_apple}'. Winning red apple '{winning_red_apple}'."
        logging.debug(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        logging.error(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        logging.debug(message)
