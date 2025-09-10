"""
Universal Sentence Encoder leveraging HuggingFace models

This module defines a constraint class that uses advanced sentence embedding models from
the HuggingFace sentence-transformers library.

It enables semantic similarity checks between original and modified sentences,
supporting both efficient and high-accuracy transformer models.

This approach is ideal for robust and flexible text augmentation and
natural language processing tasks.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder


class UniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder from HuggingFace sentence-transformers."""

    def __init__(
        self,
        threshold: float = 0.8,
        large: bool = False,
        metric: str = "angular",
        **kwargs,
    ):
        """Initialize the Universal Sentence Encoder.

        Args:
            threshold: Similarity threshold for constraint validation
            large: Whether to use the large model variant
            metric: Distance metric to use for similarity computation
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(threshold=threshold, metric=metric, **kwargs)

        # Choose model based on size preference
        if large:
            # Use a larger, more accurate model
            model_name = "sentence-transformers/all-mpnet-base-v2"
        else:
            # Use a smaller, faster model similar to USE
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self._model_name = model_name
        # Lazily load the model
        self.model = None

    def encode(self, sentences: str | list[str]) -> np.ndarray:
        """Encode sentences into embeddings.

        Args:
            sentences: Single sentence string or list of sentences to encode

        Returns:
            NumPy array of sentence embeddings
        """
        if not self.model:
            self.model = SentenceTransformer(self._model_name)

        # Ensure sentences is a list
        if isinstance(sentences, str):
            sentences = [sentences]

        # Get embeddings and convert to numpy
        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        return embeddings

    def __getstate__(self) -> dict:
        """Prepare object for pickling by removing unpicklable model."""
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore object after unpickling by resetting model to None."""
        self.__dict__ = state
        self.model = None
