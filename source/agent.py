# Description: AI agent logic for the 'Apples to Apples' game.

# Standard Libraries
import logging
import random
from enum import Enum, auto

# Third-party Libraries

# Local Modules
from apples import GreenApple, RedApple, Deck


class PlayerType(Enum):
    HUMAN = auto()
    AI = auto()
    RANDOM = auto()


# Mapping of user input to AgentType
player_type_mapping = {
    '1': PlayerType.HUMAN,
    '2': PlayerType.AI,
    '3': PlayerType.RANDOM
}


class Player:
    def __init__(self, name: str, type: PlayerType) -> None:
        self.name: str = name
        self.type: PlayerType = type
        self.points: int = 0
        self.judge: bool = False
        self.green_apple: GreenApple | None = None
        self.red_apples: list[RedApple] = []

    def __str__(self) -> str:
        return f"Player(name={self.name}, points={self.points}, judge={self.judge}, red_apples={self.red_apples})"

    def __repr__(self) -> str:
        return f"Player(name={self.name}, points={self.points}, judge={self.judge}, red_apples={self.red_apples})"

    def draw_red_apples(self, red_apple_deck: Deck) -> Deck:
        # Calculate the number of red cards to pick up
        diff = 7 - len(self.red_apples)
        if diff > 0:
            for _ in range(diff):
                # Pick up a red card
                self.red_apples.append(red_apple_deck.draw_apple())
            if diff == 1:
                print(f"{self.name} picked up 1 red card.")
                logging.info(f"{self.name} picked up 1 red card.")
            else:
                print(f"{self.name} picked up {diff} red cards.")
                logging.info(f"{self.name} picked up {diff} red cards.")
        else:
            print(f"{self.name} cannot pick up any more red cards. Player already has 7 red cards")
            logging.info(f"{self.name} cannot pick up the red card. Player already has 7 red cards")

    def draw_green_apple(self, green_apple_deck: Deck) -> GreenApple:
        # Check if the player is a judge
        if self.judge:
            self.green_apple = green_apple_deck.draw_apple()
        else:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

        # Display the green card drawn
        print(f"{self.name} drew the green card '{self.green_apple.adjective}'.")
        logging.info(f"{self.name} drew the green card '{self.green_apple}'.")

        return self.green_apple

    def choose_red_apple(self) -> RedApple:
        # Check if the player is a judge
        if self.judge:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

        # Check what type of player the player is
        red_apple: RedApple = None
        if self.type == PlayerType.HUMAN:
            # Display the red cards in the player's hand
            print(f"{self.name}'s red cards:")
            for i, red_apple in enumerate(self.red_apples):
                print(f"{i + 1}. {red_apple.noun}")

            # Prompt the player to choose a red card
            red_apple_len = len(self.red_apples)
            red_apple_index = input(f"Choose a red card(1 - {red_apple_len}): ")

            # Validate the input
            while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
                print(f"Invalid input. Please choose a valid red card (1 - {red_apple_len}).")
                red_apple_index = input("Choose a red card: ")

            # Convert the input to an index
            red_apple_index = int(red_apple_index) - 1

            # Remove the red card from the player's hand
            red_apple = self.red_apples.pop(red_apple_index)

        elif self.type == PlayerType.AI:
            pass
        elif self.type == PlayerType.RANDOM:
            red_apple = self.red_apples.pop(random.choice(range(len(self.red_apples))))
        else:
            logging.error(f"{self.type} is not a valid player type.")
            raise ValueError(f"{self.type} is not a valid player type.")

        # Display the red card chosen
        print(f"{self.name} chose a red card.")
        logging.info(f"{self.name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the player is a judge
        if not self.judge:
            logging.error(f"{self.name} is not the judge.")
            raise ValueError(f"{self.name} is not the judge.")

        # Check what type of player the player is
        winning_red_apple: dict[str, RedApple] = {}
        if self.type == PlayerType.HUMAN:
            # Display the red cards submitted by the other players
            print("Red cards submitted by the other players:")
            for i, red_apple in enumerate(red_apples):
                print(f"{i + 1}. {red_apple[list(red_apple.keys())[0]].noun}")

            # Prompt the player to choose a red card
            red_apple_len = len(red_apples)
            red_apple_index = input(f"Choose a winning red card(1 - {red_apple_len}): ")

            # Validate the input
            while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
                print(f"Invalid input. Please choose a valid red card (1 - {red_apple_len}).")
                red_apple_index = input("Choose a winning red card: ")

            # Convert the input to an index
            red_apple_index = int(red_apple_index) - 1

            # Remove the red card from the player's hand
            winning_red_apple = red_apples.pop(red_apple_index)

        elif self.type == PlayerType.AI:
            pass
        elif self.type == PlayerType.RANDOM:
            winning_red_apple = random.choice(red_apples)
        else:
            logging.error(f"{self.type} is not a valid player type.")
            raise ValueError(f"{self.type} is not a valid player type.")

        # Display the red card chosen
        logging.debug(f"winning_red_apple: {winning_red_apple}")
        round_winner = list(winning_red_apple.keys())[0]
        winning_red_apple_noun = winning_red_apple[round_winner].noun
        print(f"{self.name} chose the winning red card '{winning_red_apple_noun}'.")
        logging.info(f"{self.name} chose the winning red card '{winning_red_apple}'.")

        return winning_red_apple
