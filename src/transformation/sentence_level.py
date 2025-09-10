import random
import string
from enum import Enum

from src.transformation.base import TransformationBase


class InsertionLocation(Enum):
    """
    Enum for the location of the insertion.
    """

    START = "start"
    END = "end"
    MIDDLE = "random"
    SPREAD = "spread"


class RandomSequenceTransformation(TransformationBase):
    """
    Transformation that operates at the sentence level.
    """

    def __init__(
        self,
        num_words: int = 5,
        word_length: int = 10,
        insertion_location: InsertionLocation = InsertionLocation.END,
        num_transformations: int = 1,
    ):
        """
        Initialize the RandomSequenceTransformation.

        Args:
            num_words: Number of random words to generate
            word_length: Length of each random word in characters
            insertion_location: Where to insert the random sequence in the sentence
            num_transformations: Number of transformations to generate per input
        """
        super().__init__(num_transformations)
        self.num_words = num_words
        self.word_length = word_length
        self.insertion_location = insertion_location

    def _generate_random_word(self) -> str:
        """
        Generate a single random word with alphanumeric characters.

        Returns:
            A random string of specified length containing alphanumeric characters
        """
        characters = string.ascii_letters + string.digits
        return "".join(random.choice(characters) for _ in range(self.word_length))

    def _generate_random_sequence(self) -> str:
        """
        Generate a sequence of random words.

        Returns:
            A space-separated string of random words
        """
        words = [self._generate_random_word() for _ in range(self.num_words)]
        return " ".join(words)

    def _insert_at_start(self, sentence: str, random_sequence: str) -> str:
        """
        Insert random sequence at the start of the sentence.

        Args:
            sentence: The input sentence
            random_sequence: The random word sequence to insert

        Returns:
            The sentence with random sequence at the start
        """
        return f"{random_sequence} {sentence}"

    def _insert_at_end(self, sentence: str, random_sequence: str) -> str:
        """
        Insert random sequence at the end of the sentence.

        Args:
            sentence: The input sentence
            random_sequence: The random word sequence to insert

        Returns:
            The sentence with random sequence at the end
        """
        return f"{sentence} {random_sequence}"

    def _insert_at_middle(self, sentence: str, random_sequence: str) -> str:
        """
        Insert random sequence at a random position in the middle of the sentence.

        Args:
            sentence: The input sentence
            random_sequence: The random word sequence to insert

        Returns:
            The sentence with random sequence inserted in the middle
        """
        words = sentence.split()
        if len(words) <= 1:
            return self._insert_at_end(sentence, random_sequence)

        insert_position = random.randint(1, len(words))
        words.insert(insert_position, random_sequence)
        return " ".join(words)

    def _insert_spread(self, sentence: str) -> str:
        """
        Insert random words spread throughout the sentence at various positions.

        Args:
            sentence: The input sentence

        Returns:
            The sentence with random words spread throughout
        """
        words = sentence.split()
        if len(words) <= 1:
            random_sequence = self._generate_random_sequence()
            return self._insert_at_end(sentence, random_sequence)

        # Generate individual random words to spread
        random_words = [self._generate_random_word() for _ in range(self.num_words)]

        # Create possible insertion positions (between words)
        possible_positions = list(range(1, len(words) + 1))

        # Limit insertions to available positions
        num_insertions = min(len(random_words), len(possible_positions))

        # Randomly select positions and sort in reverse to maintain indices
        selected_positions = random.sample(possible_positions, num_insertions)
        selected_positions.sort(reverse=True)

        # Insert random words at selected positions
        for i, pos in enumerate(selected_positions):
            if i < len(random_words):
                words.insert(pos, random_words[i])

        return " ".join(words)

    def _transform_single(self, sentence: str) -> str:
        """
        Transform the sentence by inserting random words at the specified location.

        Args:
            sentence: The input sentence to transform

        Returns:
            The transformed sentence with random words inserted
        """
        if self.insertion_location == InsertionLocation.START:
            random_sequence = self._generate_random_sequence()
            return self._insert_at_start(sentence, random_sequence)
        elif self.insertion_location == InsertionLocation.END:
            random_sequence = self._generate_random_sequence()
            return self._insert_at_end(sentence, random_sequence)
        elif self.insertion_location == InsertionLocation.MIDDLE:
            random_sequence = self._generate_random_sequence()
            return self._insert_at_middle(sentence, random_sequence)
        elif self.insertion_location == InsertionLocation.SPREAD:
            return self._insert_spread(sentence)
        else:
            # Default to end if unknown location
            random_sequence = self._generate_random_sequence()
            return self._insert_at_end(sentence, random_sequence)

    def transform(self, sentence: str) -> str | list[str]:
        """
        Transform the sentence by inserting random words at the specified location.

        Args:
            sentence: The input sentence to transform

        Returns:
            Either a single transformed string (if num_transformations=1) or
            a list of transformed strings (if num_transformations>1)
        """
        if self.num_transformations == 1:
            return self._transform_single(sentence)
        else:
            return [
                self._transform_single(sentence)
                for _ in range(self.num_transformations)
            ]


if __name__ == "__main__":
    """
    Main function to test the RandomSequenceTransformation class.
    """
    transformer = RandomSequenceTransformation(
        num_transformations=10, insertion_location=InsertionLocation.SPREAD
    )
    print(transformer.transform("The quick brown fox jumps over the lazy dog."))
