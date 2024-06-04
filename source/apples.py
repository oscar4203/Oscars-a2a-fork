# Description: Green and Red 'Apple' cards.

# Standard Libraries
import random
import csv
import numpy as np

# Third-party Libraries

# Local Modules


# All apples retrieved from: http://www.com-www.com/applestoapples/
# All known sets: https://boardgamegeek.com/wiki/page/Apples_to_Apples_Series
class GreenApple:
    def __init__(self, set: str, adjective: str, synonyms: list[str] | None = None) -> None:
        self.set: str = set
        self.adjective: str = adjective
        self.synonyms: list[str] | None = synonyms
        self.adjective_vector: np.ndarray | None = None
        self.synonyms_vector: np.ndarray | None = None

    def __str__(self) -> str:
        return f"GreenApple(set={self.set}, adjective={self.adjective}, synonyms={self.synonyms}), " \
               f"adjective_vector={self.adjective_vector}, synonyms_vector={self.synonyms_vector}"

    def __repr__(self) -> str:
        return f"GreenApple(set={self.set}, adjective={self.adjective}, synonyms={self.synonyms}), " \
               f"adjective_vector={self.adjective_vector}, synonyms_vector={self.synonyms_vector}"

    def get_set(self) -> str:
        return self.set

    def get_adjective(self) -> str:
        return self.adjective

    def get_synonyms(self) -> list[str] | None:
        return self.synonyms

    def get_adjective_vector(self) -> np.ndarray | None:
        return self.adjective_vector

    def get_synonyms_vector(self) -> np.ndarray | None:
        return self.synonyms_vector

    def set_adjective_vector(self, vector: np.ndarray) -> None:
        self.adjective_vector = vector

    def set_synonyms_vector(self, vector: np.ndarray) -> None:
        self.synonyms_vector = vector


class RedApple:
    def __init__(self, set: str, noun: str, description: str | None = None) -> None:
        self.set: str = set
        self.noun: str = noun
        self.description: str | None = description
        self.noun_vector: np.ndarray | None = None
        self.description_vector: np.ndarray | None = None

    def __str__(self) -> str:
        return f"RedApple(set={self.set}, noun={self.noun}, description={self.description}), " \
               f"noun_vector={self.noun_vector}, description_vector={self.description_vector}"

    def __repr__(self) -> str:
        return f"RedApple(set={self.set}, noun={self.noun}, description={self.description}), " \
               f"noun_vector={self.noun_vector}, description_vector={self.description_vector}"

    def get_set(self) -> str:
        return self.set

    def get_noun(self) -> str:
        return self.noun

    def get_description(self) -> str | None:
        return self.description

    def get_noun_vector(self) -> np.ndarray | None:
        return self.noun_vector

    def get_description_vector(self) -> np.ndarray | None:
        return self.description_vector

    def set_noun_vector(self, vector: np.ndarray) -> None:
        self.noun_vector = vector

    def set_description_vector(self, vector: np.ndarray) -> None:
        self.description_vector = vector


class Deck:
    def __init__(self) -> None:
        self.apples: list[GreenApple | RedApple] = []

    def __str__(self) -> str:
        return f"Deck(apples={self.apples})"

    def __repr__(self) -> str:
        return f"Deck(apples={self.apples})"

    def load_deck(self, apple_type: str, filename: str) -> None:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if apple_type == "Green Apples":
                    # Extract the synomyms to a list
                    synonyms: list[str] = row["synonyms"].strip().split(", ")
                    # Append the GreenApple object to the list
                    self.apples.append(GreenApple(row["set"], row["green apples/adjectives"], synonyms))
                elif apple_type == "Red Apples":
                    # Append the RedApple object to the list
                    self.apples.append(RedApple(row["set"], row["red apples/nouns"], row["description"]))
                elif apple_type == "Green Apples Expansion":
                    # Extract the synomyms to a list
                    synonyms: list[str] = row["synonyms"].strip().split(", ")
                    # Append the GreenApple object to the list
                    self.apples.append(GreenApple(row["set"], row["adjective"], synonyms))
                elif apple_type == "Red Apples Expansion":
                    # Append the RedApple object to the list
                    self.apples.append(RedApple(row["set"], row["red apples/nouns"], row["description"]))

    def add_apple(self, apple: GreenApple | RedApple) -> None:
        self.apples.append(apple)

    def draw_apple(self) -> GreenApple | RedApple:
        return self.apples.pop()

    def shuffle(self) -> None:
        random.shuffle(self.apples)


if __name__ == "__main__":
    pass
