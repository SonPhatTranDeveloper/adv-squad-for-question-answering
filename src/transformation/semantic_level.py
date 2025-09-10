from textattack.augmentation import Augmenter
from textattack.transformations.sentence_transformations import BackTranslation

from src.transformation.base import TransformationBase


class BacktranslationTransformation(TransformationBase):
    """
    Transformation that operates at the semantic level
    It works by translating the sentence into another language and
    then back to the original language multiple times
    """

    def __init__(self, chained_back_translation: int = 5, num_transformations: int = 1):
        """
        Args:
            chained_back_translations: The number of times to chain the back
            translations.
            num_transformations: Number of transformations to generate per input.
        """
        super().__init__(num_transformations)
        self.transformation = BackTranslation(
            chained_back_translation=chained_back_translation
        )
        self.augmenter = Augmenter(self.transformation, constraints=[])

    def transform(self, sentence: str) -> str:
        return self.augmenter.augment(sentence)


if __name__ == "__main__":
    """
    Main function to test the BacktranslationTransformation class.
    """
    transformer = BacktranslationTransformation(num_transformations=10)
    print(transformer.transform("The quick brown fox jumps over the lazy dog."))
