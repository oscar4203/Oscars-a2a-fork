# Description: AI agent logic for the 'Apples to Apples' game.

# Standard Libraries
import logging
import random

# Third-party Libraries

# Local Modules
from apples import GreenApple, RedApple, Deck


class Player:
    def __init__(self, name: str, type: int) -> None:
        self.name: str = name
        self.type: int = type
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
        if self.type == 1: # Human
            pass
        elif self.type == 2: # AI
            pass
        elif self.type == 3: # Random
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
        if self.type == 1: # Human
            pass
        elif self.type == 2: # AI
            pass
        elif self.type == 3: # Random
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
