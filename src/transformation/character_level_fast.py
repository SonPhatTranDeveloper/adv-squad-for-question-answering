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
        max_attempts: int = 10,
    ):
        """
        Args:
            pct_words_to_swap: The percentage of words to swap.
            num_transformations: The number of transformations to perform
            per example.
            threshold: The threshold for the similarity constraint.
            Transformed sentence should be similar to the original sentence
            within a certain threshold.
            max_attempts: Maximum number of attempts to generate a transformation
            that meets the similarity threshold.
        """
        self.pct_words_to_swap = pct_words_to_swap
        self.num_transformations = num_transformations
        self.threshold = threshold
        self.max_attempts = max_attempts

    def _transform_single_sentence(self, sentence: str) -> str:
        """
        Apply character-level transformations to a single sentence.

        Args:
            sentence: The input sentence to transform.

        Returns:
            The transformed sentence.
        """
        original_sentence = sentence

        # If the sentence is empty or too short, return as is
        words = re.findall(r"\S+", sentence)
        if not words:
            return sentence

        # Try to create a transformation that meets the similarity threshold
        attempt = 0
        while attempt < self.max_attempts:
            current_sentence = sentence

            # Tokenize the sentence into words (preserving punctuation and spaces)
            words = re.findall(r"\S+", current_sentence)

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
            parts = re.split(r"(\s+)", current_sentence)
            word_index = 0

            for i in range(0, len(parts), 2):  # Only process non-whitespace parts
                if (
                    i < len(parts) and parts[i].strip() and word_index < len(words)
                ):  # If it's a non-empty word
                    parts[i] = words[word_index]
                    word_index += 1

            transformed_sentence = "".join(parts)

            # Check similarity constraint
            similarity_score = get_similarity(original_sentence, transformed_sentence)

            # If similarity meets the threshold, return the transformed sentence
            if similarity_score >= self.threshold:
                return transformed_sentence

            # Increment attempt counter for next iteration
            attempt += 1

        # If we've exhausted all attempts and still haven't met the threshold,
        # return the original sentence
        return original_sentence

    def transform(self, sentence: str) -> str | list[str]:
        """
        Transform the input sentence using character-level transformations.

        Args:
            sentence: The input sentence to transform.

        Returns:
            If num_transformations == 1: The transformed sentence (str).
            If num_transformations > 1: A list of transformed sentences (list[str]).
        """
        transformed_sentences = []

        for _ in range(self.num_transformations):
            transformed_sentence = self._transform_single_sentence(sentence)
            transformed_sentences.append(transformed_sentence)

        # Return single string if num_transformations == 1, otherwise return list
        if self.num_transformations == 1:
            return transformed_sentences[0]
        else:
            return transformed_sentences


if __name__ == "__main__":
    """
    Main function to test the CharacterLevelTransformation classes.
    """
    test_sentence = "The quick brown fox jumps over the lazy dog."

    # Test fast transformation with single transformation (returns string)
    fast_transformer_single = CharacterLevelTransformationFast(
        num_transformations=1, pct_words_to_swap=0.8, max_attempts=5
    )
    result_single = fast_transformer_single.transform(test_sentence)
    print(result_single)
