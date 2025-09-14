import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class ContradictionChecker:
    """
    Check if a new sentence is contradictory to a given context using OpenAI ChatGPT.

    This class uses OpenAI's GPT models to determine if a given sentence
    contradicts the information provided in a context.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Initialize the ContradictionChecker.

        Args:
            model_name: OpenAI model to use (e.g., "gpt-4o-mini")
            temperature: Temperature for the model (0.0-2.0)
        """
        self.model_name = model_name
        self.temperature = temperature

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized ContradictionChecker with model: {self.model_name}")

    def check(self, sentence: str, context: str) -> bool:
        """
        Check if a sentence contradicts the given context.

        Args:
            sentence: The sentence to check for contradiction
            context: The context to check against

        Returns:
            True if the sentence contradicts the context, False otherwise
        """
        prompt = self._create_contradiction_prompt(sentence, context)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a logical reasoning assistant that determines if statements contradict given contexts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=10,
        )

        result = response.choices[0].message.content.strip().lower()

        # Parse the response - expect "yes" or "no"
        if result.startswith("yes"):
            return True
        elif result.startswith("no"):
            return False
        else:
            raise ValueError(f"Unexpected response from OpenAI: {result}")

    def _create_contradiction_prompt(self, sentence: str, context: str) -> str:
        """
        Create a prompt for the contradiction checking task.

        Args:
            sentence: The sentence to check
            context: The context to check against

        Returns:
            The formatted prompt string
        """
        return f"""Given the following context and a sentence, determine if the sentence contradicts any information in the context.

Context:
{context}

Sentence to check:
{sentence}

Does the sentence contradict the context factually? Answer with only "yes" or "no".

Answer:"""
