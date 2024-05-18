# Description: Green and Red 'Apple' cards.

# Standard Libraries
import random

# Third-party Libraries

# Local Modules


# All apples retrieved from: http://www.com-www.com/applestoapples/
# All known sets: https://boardgamegeek.com/wiki/page/Apples_to_Apples_Series
class GreenApple:
    def __init__(self, set: str, adjective: str, synonyms: list[str] | None = None) -> None:
        self.set: str = set
        self.adjective: str = adjective
        self.synonyms: list[str] | None = synonyms

    def __str__(self) -> str:
        return f"GreenApple(set={self.set}, adjective={self.adjective}, synonyms={self.synonyms})"

    def __repr__(self) -> str:
        return f"GreenApple(set={self.set}, adjective={self.adjective}, synonyms={self.synonyms})"


class RedApple:
    def __init__(self, set: str, noun: str, description: str | None = None) -> None:
        self.set: str = set
        self.noun: str = noun
        self.description: str | None = description

    def __str__(self) -> str:
        return f"RedApple(set={self.set}, noun={self.noun}, description={self.description})"

    def __repr__(self) -> str:
        return f"RedApple(set={self.set}, noun={self.noun}, description={self.description})"


class Deck:
    def __init__(self) -> None:
        self.apples: list[GreenApple | RedApple] = []

    def __str__(self) -> str:
        return f"Deck(apples={self.apples})"

    def __repr__(self) -> str:
        return f"Deck(apples={self.apples})"

    def add_apple(self, apple: GreenApple | RedApple) -> None:
        self.apples.append(apple)

    def draw_apple(self) -> GreenApple | RedApple:
        return self.apples.pop()

    def shuffle(self) -> None:
        random.shuffle(self.apples)


if __name__ == "__main__":
    pass
