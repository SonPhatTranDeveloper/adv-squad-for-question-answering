import logging
import os

import openai

from src.transformation.base import TransformationBase

logger = logging.getLogger(__name__)


class AdversarialTransformation(TransformationBase):
    """
    Adversarial transformation that uses GPT-4o to generate distraction sentences
    to fool question answering models by adding misleading context.
    """

    def __init__(
        self,
        num_transformations: int = 1,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 150,
        temperature: float = 0.7,
        insertion_position: str = "random",  # "start", "end", "random"
    ):
        """
        Initialize the adversarial transformation.

        Args:
            num_transformations: Number of transformations to generate per input
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: GPT model to use for generating distractions
            max_tokens: Maximum tokens for GPT response
            temperature: Temperature for GPT generation (0.0-1.0)
            insertion_position: Where to insert distraction ("start", "end", "random")
        """
        super().__init__(num_transformations)

        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.insertion_position = insertion_position

        logger.info(f"Initialized AdversarialTransformation with model: {model}")

    def _generate_distraction_prompt(
        self, context: str, question: str, answer: str
    ) -> str:
        """
        Create a prompt for GPT to generate a distraction sentence.

        Args:
            context: The original context
            question: The input question
            answer: The correct answer

        Returns:
            Formatted prompt for GPT
        """
        prompt = f"""You are an expert adversarial example generator for
question-answering datasets. Your task is to produce one short, fluent sentence
that can be inserted into a passage (context) so that an automatic
QA model is more likely to produce an incorrect answer.

Given the following context and answer, generate a single distraction sentence that:
1. Is grammatically correct and fits naturally within the context's topic and style
2. Contains information that could mislead a QA model away from the correct answer
3. Avoid contradicting facts a human would immediately detect as false, but be highly
   confusable for models.
4. Matches the answer type (e.g., if the question asks for a date,
   include a date-like phrase; if it asks for a person, mention a person).
   For example, if the answer is a number, the distraction sentence should contain
   a distracting number.

Context: {context}

Question: {question}

Insert position: {self.insertion_position}

Generate ONLY the distraction sentence, no explanation or additional text:"""

        return prompt

    def _call_gpt(self, prompt: str) -> str:
        """
        Call GPT API to generate distraction sentence.

        Args:
            prompt: The formatted prompt

        Returns:
            Generated distraction sentence
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=1,
            )

            distraction = response.choices[0].message.content.strip()
            logger.debug(f"Generated distraction: {distraction}")
            return distraction

        except Exception as e:
            logger.error(f"Error calling GPT API: {e}")
            # Return empty string on error - the calling code will handle this
            return ""

    def _insert_distraction(self, context: str, distraction: str) -> str:
        """
        Insert distraction sentence into context at specified position.

        Args:
            context: Original context
            distraction: Distraction sentence to insert

        Returns:
            Modified context with distraction inserted
        """
        if not distraction.strip():
            logger.warning("Empty distraction sentence, returning original context")
            return context

        sentences = context.split(". ")

        if self.insertion_position == "start":
            # Insert at the beginning
            modified_context = f"{distraction} {context}"
        elif self.insertion_position == "end":
            # Insert at the end
            modified_context = f"{context} {distraction}"
        else:  # random
            # Insert randomly in the middle
            if len(sentences) <= 2:
                # If context is too short, insert at the end
                modified_context = f"{context} {distraction}"
            else:
                import random

                # Insert at a random position (not first or last sentence)
                insert_pos = random.randint(1, len(sentences) - 1)
                sentences.insert(insert_pos, distraction)
                modified_context = ". ".join(sentences)

        return modified_context

    def transform(self, context: str, question: str, answer: str) -> str | list[str]:
        """
        Transform the context by adding adversarial distraction sentences.

        Args:
            context: The input context to transform
            question: The input question to transform
            answer: The correct answer (used to generate relevant distractions)

        Returns:
            Either a single transformed string (if num_transformations=1) or
            a list of transformed strings (if num_transformations>1)
        """
        if not context or not answer:
            logger.warning(
                "Empty context or answer provided, returning original context"
            )
            return context if self.num_transformations == 1 else [context]

        results = []

        for i in range(self.num_transformations):
            try:
                # Generate distraction prompt
                prompt = self._generate_distraction_prompt(context, question, answer)

                # Call GPT to generate distraction
                distraction = self._call_gpt(prompt)

                # Insert distraction into context
                transformed_context = self._insert_distraction(context, distraction)
                results.append(transformed_context)

                logger.info(
                    f"Successfully generated transformation {i + 1}/"
                    f"{self.num_transformations}"
                )

            except Exception as e:
                logger.error(f"Error in transformation {i + 1}: {e}")
                # On error, add original context
                results.append(context)

        # Return single string if num_transformations=1, otherwise return list
        if self.num_transformations == 1:
            return results[0] if results else context
        else:
            return results if results else [context]
