import logging

# Import the custom tokenizer
import torch
from transformers import M2M100ForConditionalGeneration

from src.transformation.base import TransformationBase
from src.utils.tokenization_small100 import SMALL100Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktranslationTransformation(TransformationBase):
    """
    Transformation that operates at the semantic level using SMALL-100 model.
    It works by translating the sentence into another language and
    then back to the original language multiple times.
    """

    # Available intermediate languages for back translation
    INTERMEDIATE_LANGUAGES = [
        "fr",
        "de",
        "es",
        "it",
        "pt",
        "ru",
        "zh",
        "ja",
        "ko",
        "ar",
        "hi",
        "nl",
        "sv",
        "da",
        "no",
        "fi",
        "pl",
        "cs",
        "hu",
        "tr",
    ]

    def __init__(
        self,
        chained_back_translation: int = 5,
        num_transformations: int = 1,
        source_language: str = "en",
        intermediate_language: str = "fr",
        model_name: str = "alirezamsh/small100",
        max_length: int = 256,
        num_beams: int = 5,
        device: str | None = None,
    ):
        """
        Args:
            chained_back_translation: The number of times to chain the back
                translations.
            num_transformations: Number of transformations to generate per input.
            source_language: Source language code (default: "en" for English).
            intermediate_language: Single intermediate language code to use for
                back translation.
            model_name: Name of the SMALL-100 model to use.
            max_length: Maximum length for generated sequences.
            num_beams: Number of beams for beam search.
            device: Device to run the model on. If None, will auto-detect.
        """
        super().__init__(num_transformations)

        self.chained_back_translation = chained_back_translation
        self.source_language = source_language
        self.max_length = max_length
        self.num_beams = num_beams

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Set intermediate language
        if intermediate_language not in self.INTERMEDIATE_LANGUAGES:
            logger.warning(
                f"Invalid intermediate language: {intermediate_language}, "
                "using 'fr' as default"
            )
            self.intermediate_language = "fr"
        else:
            self.intermediate_language = intermediate_language

        logger.info(f"Using intermediate language: {self.intermediate_language}")

        # Load model and tokenizer
        logger.info(f"Loading SMALL-100 model: {model_name}")
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = SMALL100Tokenizer.from_pretrained(model_name)
        logger.info("SMALL-100 model loaded successfully")

    def _translate(self, text: str, target_language: str) -> str:
        """
        Translate text to target language using SMALL-100 model.

        Args:
            text: Input text to translate.
            target_language: Target language code.

        Returns:
            Translated text.
        """
        # Set target language
        self.tokenizer.tgt_lang = target_language

        # Tokenize input
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,
            )

        # Decode translation
        translated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        return translated_text.strip()

    def _back_translate_single(self, text: str) -> str:
        """
        Perform a single back translation cycle.

        Args:
            text: Input text to back translate.

        Returns:
            Back translated text.
        """
        logger.debug(
            f"Back translating via {self.intermediate_language}: {text[:50]}..."
        )

        # Translate to intermediate language
        intermediate_text = self._translate(text, self.intermediate_language)

        # Translate back to source language
        back_translated_text = self._translate(intermediate_text, self.source_language)

        logger.debug(f"Back translation result: {back_translated_text[:50]}...")

        return back_translated_text

    def _chained_back_translate(self, text: str) -> str:
        """
        Perform chained back translation.

        Args:
            text: Input text to back translate.

        Returns:
            Final back translated text after chaining.
        """
        current_text = text

        for i in range(self.chained_back_translation):
            logger.debug(
                f"Chained back translation step {i + 1}/{self.chained_back_translation}"
            )
            current_text = self._back_translate_single(current_text)

            # Early stopping if text becomes too short or empty
            if len(current_text.strip()) < 5:
                logger.warning(
                    "Back translation resulted in very short text, stopping early"
                )
                break

        return current_text

    def transform(self, context: str, question: str, answer: str) -> str | list[str]:
        """
        Transform the input context using back translation.

        Args:
            context: The input context to transform.
            question: The input question to transform.
            answer: The input answer (not used in semantic transformation).

        Returns:
            Either a single transformed string (if num_transformations=1) or
            a list of transformed strings (if num_transformations>1).
        """
        if self.num_transformations == 1:
            return self._chained_back_translate(context)
        else:
            results = []
            for i in range(self.num_transformations):
                logger.debug(
                    f"Generating transformation {i + 1}/{self.num_transformations}"
                )
                transformed = self._chained_back_translate(context)
                results.append(transformed)
            return results


if __name__ == "__main__":
    """
    Main function to test the BacktranslationTransformation class.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Test with default settings
    print("Testing SMALL-100 Back Translation...")
    transformer = BacktranslationTransformation(
        chained_back_translation=2,  # Reduced for faster testing
        num_transformations=1,
        intermediate_language="fr",  # Use French as intermediate language
    )

    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original: {test_text}")

    result = transformer.transform(test_text, "")
    print(f"Back translated: {result}")
