# Description: Green and Red 'Apple' cards.

# Standard Libraries
import random
import csv
import numpy as np
import re

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules


class Apple:
    def __init__(self, set: str) -> None:
        self.set: str = set

    def __str__(self) -> str:
        return f"Apple(set={self.set})"

    def __repr__(self) -> str:
        return f"Apple(set={self.set})"

    def get_set(self) -> str:
        return self.set

    def __format_text(self, text: str) -> str:
        # Remove the leading and trailing whitespace
        formatted_text = text.strip()

        # Remove all punctuation
        formatted_text = formatted_text.translate(str.maketrans("", "", ".,!?"))

        # Remove all special characters
        formatted_text = formatted_text.translate(str.maketrans("", "", "[](){}<>&$@#%^*+=_~`|\\/:;\""))

        # Replace all concurrent whitespaces with a single underscore
        formatted_text = re.sub(r'\s+', '_', formatted_text)

        # Convert the text to lowercase
        formatted_text = formatted_text.lower()

        return formatted_text

    def __split_text(self, text: str) -> list[str]:
        return text.split("_")


# All apples retrieved from: http://www.com-www.com/applestoapples/
# All known sets: https://boardgamegeek.com/wiki/page/Apples_to_Apples_Series
class GreenApple(Apple):
    def __init__(self, set: str, adjective: str, synonyms: list[str] | None = None) -> None:
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

    def get_adjective(self) -> str:
        return self.adjective

    def get_synonyms(self) -> list[str] | None:
        return self.synonyms

    def get_adjective_vector(self) -> np.ndarray | None:
        return self.adjective_vector

    def get_synonyms_vector(self) -> np.ndarray | None:
        return self.synonyms_vector

    def set_adjective_vector(self, model: KeyedVectors) -> None:
        cleaned_adjective = self.__format_text(self.adjective)
        self.adjective_vector = model[cleaned_adjective]

    def set_synonyms_vector(self, model: KeyedVectors) -> None:
        if self.synonyms is None:
            raise ValueError("Synonyms are not available for this Green Apple.")

        # Clean the synonyms
        cleaned_synonyms = [self.__format_text(synonym) for synonym in self.synonyms]

        # Sum the vectors for each synonym, then divide by the number of synonyms
        avg_vector = np.zeros(model.vector_size)
        for synonym in cleaned_synonyms:
            avg_vector += model[synonym]
        avg_vector /= len(cleaned_synonyms)
        self.synonyms_vector = avg_vector


class RedApple(Apple):
    def __init__(self, set: str, noun: str, description: str | None = None) -> None:
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

    def get_noun(self) -> str:
        return self.noun

    def get_description(self) -> str | None:
        return self.description

    def get_noun_vector(self) -> np.ndarray | None:
        return self.noun_vector

    def get_description_vector(self) -> np.ndarray | None:
        return self.description_vector

    def set_noun_vector(self, model: KeyedVectors) -> None:
        cleaned_noun = self.__format_text(self.noun)
        self.noun_vector = model[cleaned_noun]

    def set_description_vector(self, model: KeyedVectors) -> None:
        if self.description is None:
            raise ValueError("Description is not available for this Red Apple.")

        # Clean the description
        cleaned_description = self.__format_text(self.description)
        self.description_vector = model[cleaned_description]


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
