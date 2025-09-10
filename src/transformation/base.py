import abc


class TransformationBase(abc.ABC):
    """
    Base class for all transformations.

    Transformations are used to perturb the input sentence.
    """

    @abc.abstractmethod
    def transform(self, sentence: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")
