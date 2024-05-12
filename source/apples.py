# Description: Green and Red 'Apple' cards.

# Standard Libraries

# Third-party Libraries

# Local Modules


class GreenApple:
    def __init__(self, adjective: str, synonyms: list[str] | None = None) -> None:
        self.adjective: str = adjective
        self.synonyms: list[str] | None = synonyms

    def __str__(self) -> str:
        return f"GreenApple(adjective={self.adjective}, synonyms={self.synonyms})"


class RedApple:
    def __init__(self, noun: str, description: str | None = None) -> None:
        self.noun: str = noun
        self.description: str | None = description

    def __str__(self) -> str:
        return f"RedApple(noun={self.noun}, description={self.description})"


if __name__ == "__main__":
    pass
