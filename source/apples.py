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
        self._set: str = set

    def __str__(self) -> str:
        return f"Apple(set={self._set})"

    def __repr__(self) -> str:
        return f"Apple(set={self._set})"

    def get_set(self) -> str:
        return self._set

    def _format_text(self, text: str) -> str:
        # Remove the leading and trailing whitespace
        formatted_text = text.strip()

        # Remove all punctuation
        formatted_text = formatted_text.translate(str.maketrans("", "", ".,!?"))

        # Remove addon text (like 's)
        formatted_text = re.sub(r"'s\b", "", formatted_text)

        # Remove all special characters
        formatted_text = formatted_text.translate(str.maketrans("", "", "[](){}<>&$@#%^*+=_~`|\\/:;\""))

        # Replace all hyphens and concurrent whitespaces with a single underscore
        formatted_text = re.sub(r"[-\s]+", "_", formatted_text)

        return formatted_text

    def _split_text(self, text: str) -> list[str]:
        return text.split("_")

    def _calculate_average_vector(self, words: list[str], nlp_model: KeyedVectors) -> np.ndarray:
        # Initialize the average vector
        avg_vector = np.zeros(nlp_model.vector_size)

        # Iterate through the words
        for word in words:
            # Try searching the nlp_model for the word
            try:
                # Add the vector of the word to the average vector
                avg_vector += nlp_model[word]
            except KeyError:
                # If the word is not found, try converting it to lowercase
                try:
                    # Add the vector of the word to the average vector
                    avg_vector += nlp_model[word.lower()]
                except KeyError:
                    print(f"The word '{word}' was not found in the nlp_model.")

        # Divide the average vector by the number of words
        avg_vector /= len(words)

        return avg_vector


# All apples retrieved from: http://www.com-www.com/applestoapples/
# All known sets: https://boardgamegeek.com/wiki/page/Apples_to_Apples_Series
class GreenApple(Apple):
    def __init__(self, set: str, adjective: str, synonyms: list[str] | None = None) -> None:
        super().__init__(set)
        self.__adjective: str = adjective
        self.__synonyms: list[str] | None = synonyms
        self.__adjective_vector: np.ndarray | None = None
        self.__synonyms_vector: np.ndarray | None = None

    def __str__(self) -> str:
        synonyms_str = ', '.join(self.__synonyms) if self.__synonyms is not None else None
        return f"{self.__adjective} | Synonyms: {synonyms_str}"

    def __repr__(self) -> str:
        return f"GreenApple(set={self._set}, adjective={self.__adjective}, synonyms={self.__synonyms}), " \
                f"adjective_vector={self.__adjective_vector}, synonyms_vector={self.__synonyms_vector}"

    def get_adjective(self) -> str:
        return self.__adjective

    def get_synonyms(self) -> list[str] | None:
        return self.__synonyms

    def get_adjective_vector(self) -> np.ndarray | None:
        return self.__adjective_vector

    def get_synonyms_vector(self) -> np.ndarray | None:
        return self.__synonyms_vector

    def set_adjective_vector(self, nlp_model: KeyedVectors) -> None:
        # Clean the adjective
        cleaned_adjective = self._format_text(self.__adjective)

        # Split the cleaned adjective into a list of words
        split_cleaned_adjective = self._split_text(cleaned_adjective)

        # Calculate the average vector for the adjective
        self.__adjective_vector = self._calculate_average_vector(split_cleaned_adjective, nlp_model)

    def set_synonyms_vector(self, nlp_model: KeyedVectors) -> None:
        # Check if synonyms exist
        if self.__synonyms is None:
            raise ValueError("Synonyms are not available for this Green Apple.")

        # Clean the synonyms
        cleaned_synonyms = [self._format_text(synonym) for synonym in self.__synonyms]

        # Calculate the average vector for the synonyms
        self.__synonyms_vector = self._calculate_average_vector(cleaned_synonyms, nlp_model)


class RedApple(Apple):
    def __init__(self, set: str, noun: str, description: str | None = None) -> None:
        super().__init__(set)
        self.__noun: str = noun
        self.__description: str | None = description
        self.__noun_vector: np.ndarray | None = None
        self.__description_vector: np.ndarray | None = None

    def __str__(self) -> str:
        return f"{self.__noun} | Description: {self.__description}"

    def __repr__(self) -> str:
        return f"RedApple(set={self._set}, noun={self.__noun}, description={self.__description}), " \
                f"noun_vector={self.__noun_vector}, description_vector={self.__description_vector}"

    def get_noun(self) -> str:
        return self.__noun

    def get_description(self) -> str | None:
        return self.__description

    def get_noun_vector(self) -> np.ndarray | None:
        return self.__noun_vector

    def get_description_vector(self) -> np.ndarray | None:
        return self.__description_vector

    def set_noun_vector(self, nlp_model: KeyedVectors) -> None:
        # Clean the noun
        cleaned_noun = self._format_text(self.__noun)

        # Split the cleaned noun into a list of words
        split_cleaned_noun = self._split_text(cleaned_noun)

        # Calculate the average vector for the noun
        self.__noun_vector = self._calculate_average_vector(split_cleaned_noun, nlp_model)

    def set_description_vector(self, nlp_model: KeyedVectors) -> None:
        # Check if description exists
        if self.__description is None:
            raise ValueError("Description is not available for this Red Apple.")

        # Clean the description
        cleaned_description = self._format_text(self.__description)

        # Split the cleaned description into a list of words
        split_cleaned_description = self._split_text(cleaned_description)

        # Calculate the average vector for the description
        self.__description_vector = self._calculate_average_vector(split_cleaned_description, nlp_model)


class Deck:
    def __init__(self) -> None:
        self.__apples: list[GreenApple | RedApple] = []

    def __str__(self) -> str:
        return f"Deck(apples={self.__apples})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_apples(self) -> list[GreenApple | RedApple]:
        return self.__apples

    def load_deck(self, apple_type: str, filename: str) -> None:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if apple_type == "Green Apples":
                    # Extract the synomyms to a list
                    synonyms: list[str] = row["synonyms"].strip().split(", ")
                    # Append the GreenApple object to the list
                    self.__apples.append(GreenApple(row["set"], row["green apples/adjectives"], synonyms))
                elif apple_type == "Red Apples":
                    # Append the RedApple object to the list
                    self.__apples.append(RedApple(row["set"], row["red apples/nouns"], row["description"]))
                elif apple_type == "Green Apples Expansion":
                    # Extract the synomyms to a list
                    synonyms: list[str] = row["synonyms"].strip().split(", ")
                    # Append the GreenApple object to the list
                    self.__apples.append(GreenApple(row["set"], row["adjective"], synonyms))
                elif apple_type == "Red Apples Expansion":
                    # Append the RedApple object to the list
                    self.__apples.append(RedApple(row["set"], row["red apples/nouns"], row["description"]))

    def add_apple(self, apple: GreenApple | RedApple) -> None:
        self.__apples.append(apple)

    def draw_apple(self) -> GreenApple | RedApple:
        return self.__apples.pop()

    def shuffle(self) -> None:
        random.shuffle(self.__apples)


if __name__ == "__main__":
    pass
