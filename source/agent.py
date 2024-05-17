# Description: AI agent logic for the 'Apples to Apples' game.

# Standard Libraries
import logging
import random

# Third-party Libraries

# Local Modules
from apples import GreenApple, RedApple


class Player:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.points: int = 0
        self.judge: bool = False
        self.green_apple: GreenApple | None = None
        self.red_apples: list[RedApple] = []

    def __str__(self) -> str:
        return f"Player(name={self.name}, points={self.points}, judge={self.judge}, red_apples={self.red_apples})"

    def __repr__(self) -> str:
        return f"Player(name={self.name}, points={self.points}, judge={self.judge}, red_apples={self.red_apples})"

    def pickup_red_apples(self) -> None:
        diff = 7 - len(self.red_apples)
        if diff > 0:
            for _ in range(diff):
                self.red_apples.append(RedApple("Red Card"))
            print(f"{self.name} picked up {diff} red cards.")
            logging.info(f"{self.name} picked up {diff} red cards.")
        else:
            print(f"{self.name} cannot pick up any more red cards. Player already has 7 red cards")
            logging.info(f"{self.name} cannot pick up the red card. Player already has 7 red cards")

    def choose_green_apple(self) -> GreenApple:
        # Check if the player is a judge
        if self.judge:
            ### Logic to choose a green card goes here ###
            self.green_apple = GreenApple("Green Card")
            print(f"{self.name} chose the green card '{self.green_apple.adjective}'.")
            logging.info(f"{self.name} chose the green card '{self.green_apple}'.")

            return self.green_apple

        else:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

    def choose_red_apple(self) -> RedApple:
        # Choose a red card
        # Logic to choose a red card goes here #
        red_apple = self.red_apples.pop(random.choice(range(len(self.red_apples))))
        print(f"{self.name} chose the red card '{red_apple}'.")
        logging.info(f"{self.name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the player is a judge
        if not self.judge:
            logging.error(f"{self.name} is not the judge.")
            raise ValueError(f"{self.name} is not the judge.")

        # Choose the winning red card
        # Logic to choose the winning red card goes here #
        winning_red_apple = random.choice(red_apples)
        print(f"{self.name} chose the winning red card '{winning_red_apple}'.")
        logging.info(f"{self.name} chose the winning red card '{winning_red_apple}'.")

        return winning_red_apple
