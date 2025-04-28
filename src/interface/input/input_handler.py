# Description: Abstract input handling for Apples to Apples game.

# Standard Libraries
from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING

# Third-party Libraries

# Local Modules

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple


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
