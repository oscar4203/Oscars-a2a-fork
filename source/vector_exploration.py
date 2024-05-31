# Description: Green and Red 'Apple' cards.

# Standard Libraries

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules


class Word:
    def __init__(self, word: str, meaning: str) -> None:
        self.word: str = word
        self.meaning: str = meaning
        self.vector: list[float] = []

    def __str__(self) -> str:
        return f"Word(word={self.word}, meaning={self.meaning}, vector={self.vector})"

    def __repr__(self) -> str:
        return f"Word(word={self.word}, meaning={self.meaning}, vector={self.vector})"

    def set_vector(self, vector: list[float]) -> None:
        self.vector = vector

    def get_word(self) -> str:
        return self.word

    def get_meaning(self) -> str:
        return self.meaning

    def get_vector(self) -> list[float]:
        return self.vector


class Tone:
    def __init__(self, tone: str, meaning: str, words: list[Word]) -> None:
        self.tone: str = tone
        self.meaning: str = meaning
        self.words: list[Word] = words

    def __str__(self) -> str:
        return f"Tone(tone={self.tone}, meaning={self.meaning}, words={[word.word for word in self.words]})"

    def __repr__(self) -> str:
        return f"Tone(tone={self.tone}, meaning={self.meaning}, words={[word.word for word in self.words]})"

    def add_word(self, word: Word) -> None:
        self.words.append(word)

    def remove_word(self, word: Word) -> None:
        self.words.remove(word)

    def get_tone(self) -> str:
        return self.tone

    def get_meaning(self) -> str:
        return self.meaning

    def get_words(self) -> list[Word]:
        return self.words


class Category:
    def __init__(self, category: str, meaning: str, synonyms: list[Word], antonyms: list[Word], tones: dict[str, Tone]) -> None:
        self.category: str = category
        self.meaning: str = meaning
        self.synonyms: list[Word] = synonyms
        self.antonyms: list[Word] = antonyms
        self.tones: dict[str, Tone] = tones

    def __str__(self) -> str:
        return f"Category(category={self.category}, meaning={self.meaning},
                 synonyms={self.synonyms}, antonyms={self.antonyms}, tones={[tone.tone for tone in self.tones]})"

    def __repr__(self) -> str:
        return f"Category(category={self.category}, meaning={self.meaning},
                 synonyms={self.synonyms}, antonyms={self.antonyms}, tones={[tone.tone for tone in self.tones]})"

    def add_synonym(self, synonym: Word) -> None:
        self.synonyms.append(synonym)

    def remove_synonym(self, synonym: Word) -> None:
        self.synonyms.remove(synonym)

    def add_tone(self, tone: Tone) -> None:
        self.tones[tone.tone] = tone

    def remove_tone(self, tone: Tone) -> None:
        self.tones.pop(tone.tone)

    def get_category(self) -> str:
        return self.category

    def get_meaning(self) -> str:
        return self.meaning

    def get_synonyms(self) -> list[Word]:
        return self.synonyms

    def get_antonyms(self) -> list[Word]:
        return self.antonyms

    def get_tones(self) -> list[Tone]:
        return list(self.tones.values())


# Define all the tones
tone_words = {
    "positive": Tone("positive", "Words that have a positive connotation", []),
    "negative": Tone("negative", "Words that have a negative connotation", []),
    "neutral": Tone("neutral", "Words that have a neutral connotation", [])
}


# Define all the categories for the words
word_meanings = {
    "size": Category(
        "size",
        "The physical dimensions of an object",
        [ # Synonyms
            Word("dimension", ""),
            Word("scale", ""),
            Word("volume", "")
        ],
        [],
        tone_words
    ),
    "color": Category(
        "color",
        "The visual appearance of an object",
        [],
        [],
        tone_words
    ),
    "temperature": Category(
        "temperature",
        "The degree of heat present in an object",
        [], [], tone_words),
    "speed": Category(
        "speed",
        "The rate at which an object moves",
        [],
        [],
        tone_words
    ),
    "difficulty": Category(
        "difficulty",
        "The level of complexity in a task",
        [],
        [],
        tone_words
    ),
    "quality": Category(
        "quality",
        "The standard of excellence in an object",
        [],
        [],
        tone_words)
}

# Define all the words for the different tones of size
word_meanings["size"].tones["positive"].add_word(Word("big", ""))
word_meanings["size"].tones["positive"].add_word(Word("large", ""))
word_meanings["size"].tones["positive"].add_word(Word("huge", ""))
word_meanings["size"].tones["negative"].add_word(Word("small", ""))
word_meanings["size"].tones["negative"].add_word(Word("tiny", ""))
word_meanings["size"].tones["negative"].add_word(Word("minuscule", ""))
word_meanings["size"].tones["neutral"].add_word(Word("medium", ""))
word_meanings["size"].tones["neutral"].add_word(Word("average", ""))
word_meanings["size"].tones["neutral"].add_word(Word("moderate", ""))

# Define all the words for the different tones of color
word_meanings["color"].tones["neutral"].add_word(Word("red", ""))
word_meanings["color"].tones["neutral"].add_word(Word("orange", ""))
word_meanings["color"].tones["neutral"].add_word(Word("yellow", ""))
word_meanings["color"].tones["neutral"].add_word(Word("green", ""))
word_meanings["color"].tones["neutral"].add_word(Word("blue", ""))
word_meanings["color"].tones["neutral"].add_word(Word("purple", ""))
word_meanings["color"].tones["neutral"].add_word(Word("black", ""))
word_meanings["color"].tones["neutral"].add_word(Word("white", ""))

# Define all the words for the different tones of temperature
word_meanings["temperature"].tones["positive"].add_word(Word("hot", ""))
word_meanings["temperature"].tones["positive"].add_word(Word("warm", ""))
word_meanings["temperature"].tones["negative"].add_word(Word("cold", ""))
word_meanings["temperature"].tones["negative"].add_word(Word("chilly", ""))
word_meanings["temperature"].tones["negative"].add_word(Word("freezing", ""))
word_meanings["temperature"].tones["neutral"].add_word(Word("lukewarm", ""))
word_meanings["temperature"].tones["neutral"].add_word(Word("cool", ""))
word_meanings["temperature"].tones["neutral"].add_word(Word("temperate", ""))

# Define all the words for the different tones of speed
word_meanings["speed"].tones["positive"].add_word(Word("fast", ""))
word_meanings["speed"].tones["positive"].add_word(Word("quick", ""))
word_meanings["speed"].tones["positive"].add_word(Word("swift", ""))
word_meanings["speed"].tones["positive"].add_word(Word("speedy", ""))
word_meanings["speed"].tones["positive"].add_word(Word("rapid", ""))
word_meanings["speed"].tones["negative"].add_word(Word("slow", ""))
word_meanings["speed"].tones["negative"].add_word(Word("sluggish", ""))
word_meanings["speed"].tones["negative"].add_word(Word("delayed", ""))
word_meanings["speed"].tones["neutral"].add_word(Word("moderate", ""))
word_meanings["speed"].tones["neutral"].add_word(Word("average", ""))
word_meanings["speed"].tones["neutral"].add_word(Word("standard", ""))

# Define all the words for the different tones of difficulty
word_meanings["difficulty"].tones["positive"].add_word(Word("easy", ""))
word_meanings["difficulty"].tones["positive"].add_word(Word("simple", ""))
word_meanings["difficulty"].tones["positive"].add_word(Word("straightforward", ""))
word_meanings["difficulty"].tones["positive"].add_word(Word("effortless", ""))
word_meanings["difficulty"].tones["negative"].add_word(Word("hard", ""))
word_meanings["difficulty"].tones["negative"].add_word(Word("challenging", ""))
word_meanings["difficulty"].tones["negative"].add_word(Word("tough", ""))
word_meanings["difficulty"].tones["negative"].add_word(Word("difficult", ""))
word_meanings["difficulty"].tones["neutral"].add_word(Word("moderate", ""))
word_meanings["difficulty"].tones["neutral"].add_word(Word("average", ""))

# Define all the words for the different tones of quality
word_meanings["quality"].tones["positive"].add_word(Word("excellent", ""))
word_meanings["quality"].tones["positive"].add_word(Word("superior", ""))
word_meanings["quality"].tones["positive"].add_word(Word("premium", ""))
word_meanings["quality"].tones["positive"].add_word(Word("high-quality", ""))
word_meanings["quality"].tones["negative"].add_word(Word("poor", ""))
word_meanings["quality"].tones["negative"].add_word(Word("inferior", ""))
word_meanings["quality"].tones["negative"].add_word(Word("substandard", ""))
word_meanings["quality"].tones["negative"].add_word(Word("low-quality", ""))
word_meanings["quality"].tones["neutral"].add_word(Word("average", ""))
word_meanings["quality"].tones["neutral"].add_word(Word("standard", ""))
word_meanings["quality"].tones["neutral"].add_word(Word("typical", ""))

# Load vectors directly from the Google News dataset
model = KeyedVectors.load_word2vec_format("../apples/GoogleNews-vectors-negative300.bin", binary=True)

# see the shape of the vector (300,)
print(word_meanings["size"].tones["positive"].words[0].get_word())
