import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class AnswerabilityChecker:
    """
    Check if a sentence can be used to answer a given question using OpenAI GPT models.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Initialize the AnswerabilityChecker.

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
        logger.info(f"Initialized AnswerabilityChecker with model: {self.model_name}")

    def check(self, sentence: str, question: str) -> bool:
        """
        Check if a sentence can be used to answer the question.

        Args:
            sentence: The sentence to check
            question: The question being asked
        Returns:
            True if the sentence can answer the question, False otherwise
        """
        prompt = self._create_answerability_prompt(sentence, question)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a logical reasoning assistant that determines if a sentence can be used to answer a question.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=20,
        )

        result = response.choices[0].message.content.strip().lower()

        # Parse the response - expect "yes" or "no"
        if result.startswith("yes"):
            return True
        elif result.startswith("no"):
            return False
        else:
            raise ValueError(f"Unexpected response from OpenAI: {result}")

    def _create_answerability_prompt(self, sentence: str, question: str) -> str:
        """
        Create a prompt for the answerability checking task.

        Args:
            sentence: The sentence to check
            question: The question being asked
        Returns:
            The formatted prompt string
        """
        return f"""You are a logical reasoning assistant.

Task:
Determine whether the given sentence contains enough information to answer the question.

Important rules:
- The sentence should provide a direct and explicit answer to the question.
- The answer must be clearly stated in the sentence, not implied or inferred.
- Do not make assumptions or draw conclusions beyond what is explicitly written.

### Examples ###

Question:
What is the capital of France?

Sentence:
Paris is the capital of France.

Correct Output:
yes

(since the sentence directly answers the question)

---

Question:
What is the longest river in the world?

Sentence:
The Amazon River flows through South America.

Correct Output:
no

(since the sentence does not provide information about which river is longest)

---

Question:
Who developed the theory of relativity?

Sentence:
Albert Einstein developed the theory of relativity.

Correct Output:
yes

(since the sentence directly answers who developed the theory)

---

Question:
When was the Great Wall of China built?

Sentence:
The Great Wall of China was built to protect Chinese states from invasions.

Correct Output:
no

(since the sentence explains why it was built, not when)

---

Question:
How tall is Mount Everest?

Sentence:
Mount Everest is 8,848 meters tall.

Correct Output:
yes

(since the sentence provides the height information requested)

---

### Now your turn ###

Question:
{question}

Sentence:
{sentence}

Final instruction:
Can the sentence be used to answer the question?
Reply with ONLY one word: "yes" or "no".
"""
