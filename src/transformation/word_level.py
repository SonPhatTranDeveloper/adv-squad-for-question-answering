import transformers
from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import (
    CompositeTransformation,
    WordInsertionMaskedLM,
    WordMergeMaskedLM,
    WordSwapMaskedLM,
)

from src.transformation.base import TransformationBase
from src.utils.sentence_encoder import UniversalSentenceEncoder


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

    def __init__(
        self,
        model: str = "distilroberta-base",
        tokenizer: str = "distilroberta-base",
        max_candidates: int = 50,
        min_confidence_swap: float = 5e-4,
        min_confidence_insert: float = 0.0,
        min_confidence_merge: float = 5e-3,
        threshold: float = 0.7,
        num_transformations: int = 1,
    ):
        """
        Args:
            model: The model to use for the masked language model.
            tokenizer: The tokenizer to use for the masked language model.
            max_candidates: The maximum number of candidates to consider for each word.
            min_confidence_swap: The minimum confidence score for the swap.
            min_confidence_insert: The minimum confidence score for the insert.
            min_confidence_merge: The minimum confidence score for the merge.
            threshold: The threshold for the universal sentence encoder constraint.
            num_transformations: Number of transformations to generate per input.
        """
        super().__init__(num_transformations)
        shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(model)
        shared_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

        transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=max_candidates,
                    min_confidence=min_confidence_swap,
                ),
                WordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=max_candidates,
                    min_confidence=min_confidence_insert,
                ),
                WordMergeMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=max_candidates,
                    min_confidence=min_confidence_merge,
                ),
            ]
        )

        use_constraint = UniversalSentenceEncoder(
            threshold=threshold,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )

        constraints = [
            RepeatModification(),
            StopwordModification(),
            use_constraint,
        ]

        self.augmenter = Augmenter(
            transformation=transformation,
            constraints=constraints,
            transformations_per_example=self.num_transformations,
        )

    def transform(self, sentence: str) -> str:
        return self.augmenter.augment(sentence)


if __name__ == "__main__":
    """
    Main function to test the CLARETransformation class.
    """
    transformer = CLARETransformation(num_transformations=10)
    print(transformer.transform("The quick brown fox jumps over the lazy dog."))
