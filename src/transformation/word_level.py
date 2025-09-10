from textattack.augmentation.recipes import CLAREAugmenter

from src.transformation.base import TransformationBase


class CLARETransformation(TransformationBase):
    """
    Transformation that operates at the word level.

    References:
        - Li, Zhang, Peng, Chen, Brockett, Sun, Dolan.
        - “Contextualized Perturbation for Textual Adversarial Attack” (Li et al., 2020)
        - https://arxiv.org/abs/2009.07502

    CLARE builds on a pre-trained masked language model and modifies the inputs
    in a context-aware manner.
    We propose three contextualized perturbations, Replace, Insert and Merge,
    allowing for generating outputs of varied lengths.
    """

    def __init__(self):
        self.augmenter = CLAREAugmenter()

    def transform(self, sentence: str) -> str:
        return self.augmenter.augment(sentence)
