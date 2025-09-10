from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from src.transformation.base import TransformationBase
from src.utils.sentence_encoder import UniversalSentenceEncoder


class CharacterLevelTransformation(TransformationBase):
    """
    Transformation that operates at the character level.

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

    def __init__(self, pct_words_to_swap: float = 0.5, num_transformations: int = 1):
        """
        Args:
            pct_words_to_swap: The percentage of words to swap.
            num_transformations: The number of transformations to perform
            per example.
        """
        self.pct_words_to_swap = pct_words_to_swap
        self.num_transformations = num_transformations

        self.transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        self.constraints = [
            RepeatModification(),
            StopwordModification(),
            # UniversalSentenceEncoder(threshold=0.8),
        ]

        self.augmenter = Augmenter(
            self.transformation,
            self.constraints,
            pct_words_to_swap=self.pct_words_to_swap,
            transformations_per_example=self.num_transformations,
        )

    def transform(self, sentence: str) -> str:
        return self.augmenter.augment(sentence)


if __name__ == "__main__":
    """
    Main function to test the CharacterLevelTransformation class.
    """
    transformer = CharacterLevelTransformation(num_transformations=10)
    print(transformer.transform("The quick brown fox jumps over the lazy dog."))
