from textattack.augmentation import Augmenter
from textattack.transformations.sentence_transformations import BackTranslation

from src.transformation.base import TransformationBase


class BacktranslationTransformation(TransformationBase):
    """
    Transformation that operates at the semantic level
    It works by translating the sentence into another language and
    then back to the original language multiple times
    """

    def __init__(self, chained_back_translations: int = 5):
        """
        Args:
            chained_back_translations: The number of times to chain the back
            translations.
        """
        self.transformation = BackTranslation(
            chained_back_translations=chained_back_translations
        )
        self.augmenter = Augmenter(self.transformation, constraints=[])

    def transform(self, sentence: str) -> str:
        return self.augmenter.augment(sentence)
