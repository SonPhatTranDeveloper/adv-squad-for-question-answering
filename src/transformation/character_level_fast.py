import random
import re

from src.transformation.base import TransformationBase
from src.utils.similarity import get_similarity
from src.utils.strings import (
    randomly_delete_character,
    randomly_inject_space,
    randomly_substitute_character,
    randomly_substitute_word,
    randomly_swap_characters,
)


class CharacterLevelTransformationFast(TransformationBase):
    """
    Transformation that operates at the character level.

    Note:
        The words which have been modified by previous transformation will
        not be modified again.

    Consisting of:
        - Inserting space into the sentence.
        - Deleting a character from the sentence.
        - Swapping two characters in the sentence.
        - Substituting a character with another character (that is visually similar).
        - Swap words with those with similar embedding.

    References:
        Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).
        TextBugger: Generating Adversarial Text Against Real-world Applications.
        https://arxiv.org/abs/1812.05271
    """

    def __init__(
        self,
        pct_words_to_swap: float = 0.5,
        num_transformations: int = 1,
        threshold: float = 0.8,
    ):
        """
        Args:
            pct_words_to_swap: The percentage of words to swap.
            num_transformations: The number of transformations to perform
            per example.
            threshold: The threshold for the similarity constraint.
            Transformed sentence should be similar to the original sentence
            within a certain threshold.
        """
        self.pct_words_to_swap = pct_words_to_swap
        self.num_transformations = num_transformations
        self.threshold = threshold

    def transform(self, sentence: str) -> str:
        """
        Transform the input sentence using character-level transformations.

        Args:
            sentence: The input sentence to transform.

        Returns:
            The transformed sentence.
        """
        original_sentence = sentence

        for _ in range(self.num_transformations):
            # Tokenize the sentence into words (preserving punctuation and spaces)
            words = re.findall(r"\S+", sentence)

            if not words:
                continue

            # Calculate number of words to transform
            num_words_to_transform = max(1, int(len(words) * self.pct_words_to_swap))

            # Select random words to transform (without replacement)
            words_to_transform_indices = random.sample(
                range(len(words)), min(num_words_to_transform, len(words))
            )

            # Keep track of which words have been modified to avoid re-modification
            modified_words = set()

            # Apply transformations
            for word_idx in words_to_transform_indices:
                if word_idx in modified_words:
                    continue

                original_word = words[word_idx]

                # Skip if word is too short or not alphabetic for most transformations
                if len(original_word) <= 2:
                    continue

                # Randomly choose a transformation type
                transformation_type = random.choice(
                    [
                        "inject_space",
                        "delete_character",
                        "swap_characters",
                        "substitute_character",
                        "substitute_word",
                    ]
                )

                # Apply the chosen transformation
                if transformation_type == "inject_space":
                    words[word_idx] = randomly_inject_space(original_word)
                elif transformation_type == "delete_character":
                    words[word_idx] = randomly_delete_character(original_word)
                elif transformation_type == "swap_characters":
                    words[word_idx] = randomly_swap_characters(original_word)
                elif transformation_type == "substitute_character":
                    words[word_idx] = randomly_substitute_character(original_word)
                elif transformation_type == "substitute_word":
                    words[word_idx] = randomly_substitute_word(original_word)

                # Mark this word as modified
                modified_words.add(word_idx)

            # Reconstruct the sentence
            # We need to preserve the original spacing, so we'll use a
            # different approach
            # Split by whitespace but keep track of the separators
            parts = re.split(r"(\s+)", sentence)
            word_index = 0

            for i in range(0, len(parts), 2):  # Only process non-whitespace parts
                if (
                    i < len(parts) and parts[i].strip() and word_index < len(words)
                ):  # If it's a non-empty word
                    parts[i] = words[word_index]
                    word_index += 1

            sentence = "".join(parts)

            # Check similarity constraint
            try:
                similarity_score = get_similarity(original_sentence, sentence)[0][0]
                if similarity_score < self.threshold:
                    # If similarity is too low, revert to previous sentence
                    sentence = original_sentence
                    break
            except Exception:
                # If similarity calculation fails, continue with transformation
                pass

        return sentence


if __name__ == "__main__":
    """
    Main function to test the CharacterLevelTransformation classes.
    """
    # Test fast transformation
    fast_transformer = CharacterLevelTransformationFast(
        num_transformations=5, pct_words_to_swap=0.3
    )
    print("\nFast transformation:")
    print(fast_transformer.transform("The quick brown fox jumps over the lazy dog."))
