import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class ContradictionChecker:
    """
    Check if a new sentence would change the correct answer to a given question,
    given the original context and answer, using OpenAI GPT models.
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

    def check(self, context: str, question: str, answer: str, sentence: str) -> bool:
        """
        Check if a sentence would change the correct answer to the question.

        Args:
            context: The context to check against
            question: The question being asked
            answer: The current correct answer
            sentence: The sentence to check
        Returns:
            True if the sentence changes the answer, False otherwise
        """
        prompt = self._create_answer_change_prompt(context, question, answer, sentence)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a logical reasoning assistant that determines if a sentence would change a question's answer.",
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

    def _create_answer_change_prompt(
        self, context: str, question: str, answer: str, sentence: str
    ) -> str:
        """
        Create a prompt for the answer-change checking task.

        Args:
            context: The context to check against
            question: The question being asked
            answer: The current correct answer
            sentence: The sentence to check
        Returns:
            The formatted prompt string
        """
        return f"""You are a logical reasoning assistant.

Task:
Determine whether the given sentence, if added to the context, would change the correct answer to the question.

Important rules:
- Ignore minor stylistic differences, synonyms, or paraphrasing.
- Focus only on whether the new sentence would cause a different answer.
- Do not over-interpret or make assumptions beyond what is explicitly written.

### Examples ###

Context:
Paris is the capital of France.
Question:
What is the capital of France?
Original Answer:
Paris

Sentence:
London is the capital of France.

Correct Output:
yes

(since the added sentence changes the answer to the question)

---

Context:
The Amazon River is the second longest river in the world, after the Nile.
Question:
What is the longest river in the world?
Original Answer:
Nile

Sentence:
The Amazon River flows through South America.

Correct Output:
no

(since the added sentence does not change the answer to the question)

---

Context:
Albert Einstein developed the theory of relativity.
Question:
Who developed the theory of relativity?
Original Answer:
Albert Einstein

Sentence:
Einstein was born in Germany.

Correct Output:
no

(since the added sentence does not change the answer to the question)

---

Context:
The Great Wall of China was built to protect Chinese states from invasions.
Question:
Where was the Great Wall of China built?
Original Answer:
China

Sentence:
The Great Wall of China is located in India.

Correct Output:
yes

(since the added sentence changes the answer to the question)

---

### Now your turn ###

Context:
{context}

Question:
{question}

Original Answer:
{answer}

Sentence:
{sentence}

Final instruction:
Would the sentence change the correct answer to the question?
Reply with ONLY one word: "yes" or "no".
"""
