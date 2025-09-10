import abc


class TransformationBase(abc.ABC):
    """
    Base class for all transformations.

    Transformations are used to perturb the input sentence.
    """

    def __init__(self, num_transformations: int = 1):
        """
        Initialize the transformation.

        Args:
            num_transformations: Number of transformations to generate per input
        """
        self.num_transformations = num_transformations

    @abc.abstractmethod
    def transform(self, sentence: str) -> str | list[str]:
        """
        Transform the input sentence.

        Args:
            sentence: The input sentence to transform

        Returns:
            Either a single transformed string (if num_transformations=1) or
            a list of transformed strings (if num_transformations>1)
        """
        raise NotImplementedError("Subclasses must implement this method")
