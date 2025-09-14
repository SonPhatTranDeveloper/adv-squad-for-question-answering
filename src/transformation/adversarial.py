import logging
import os
import random

import openai

from src.transformation.base import TransformationBase
from src.utils.caching import persistent_cache
from src.utils.contradiction import ContradictionChecker

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
        contradiction_checker: ContradictionChecker = None,
        max_contradiction_attempts: int = 5,
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
            contradiction_checker: ContradictionChecker instance to verify non-contradictory distractions
            max_contradiction_attempts: Maximum attempts to generate non-contradictory distractions
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
        self.contradiction_checker = contradiction_checker
        self.max_contradiction_attempts = max_contradiction_attempts
        logger.info(f"Initialized AdversarialTransformation with model: {model}")

        # Initialize default contradiction checker if none provided
        if self.contradiction_checker is None:
            logger.info(
                "No contradiction checker provided, initializing default ContradictionChecker"
            )
            self.contradiction_checker = ContradictionChecker()

    def _generate_perturbed_question_prompt(self, question: str, answer: str) -> str:
        """
        Create a prompt for GPT to generate a perturbed question (Step 1).

        Args:
            question: The original question
            answer: The correct answer

        Returns:
            Formatted prompt for GPT to perturb the question
        """
        prompt = f"""You are an expert adversarial example generator for QA datasets. Your task is to perturb a question by changing key entities while keeping the same question structure and type.

Instructions:
- Perturb ONLY ONE key entity (especially proper nouns like *NAMES*, *DATES*, *NUMBERS*, *LOCATIONS*, *ORGANIZATIONS*, etc.) in the question
- Choose the most IMPORTANT key entity to perturb
- Use plausible alternatives (synonyms, antonyms, or nearby entities)
- Keep the same question type and structure
- The perturbed question should be grammatically correct and natural
- Return ONLY the perturbed question (no explanations)

Examples:

Original Question: "Where is the headquarters of the United Nations located?"
Perturbed Question: "Where is the main office of UNESCO located?"

Original Question: "Who invented the telephone?"
Perturbed Question: "Who created the first practical telegraph system?"

Original Question: "When was the first manned moon landing?"
Perturbed Question: "When did the first unmanned lunar probe land on the moon?"

Original Question: "Which company developed the first commercially successful personal computer?"
Perturbed Question: "Which corporation introduced the first popular desktop computer?"

Original Question: "What year did the French Revolution begin?"
Perturbed Question: "In which year did the American Revolution start?"

Now perturb the following question:

Original Question: {question}
Original Answer: {answer}

Perturbed Question:"""

        return prompt

    def _generate_fake_answer_prompt(
        self, perturbed_question: str, original_answer: str
    ) -> str:
        """
        Create a prompt for GPT to generate a fake answer (Step 2).

        Args:
            perturbed_question: The perturbed question from step 1
            original_answer: The original correct answer

        Returns:
            Formatted prompt for GPT to generate a fake answer
        """
        prompt = f"""You are an expert adversarial example generator for QA datasets. Your task is to create a fake answer that corresponds to the perturbed question.

Instructions:
- Create a FakeAnswer of the same type as the original Answer (location, person, date, organization, etc.)
- The FakeAnswer should have NO token overlap with the original Answer
- The FakeAnswer should be plausible and realistic for the perturbed question
- Return ONLY the fake answer (no explanations)

Examples:

Perturbed Question: "Where is the main office of UNESCO located?"
Original Answer: "New York"
Fake Answer: "Paris"

Perturbed Question: "Who created the first practical telegraph system?"
Original Answer: "Alexander Graham Bell"
Fake Answer: "Samuel Morse"

Perturbed Question: "When did the first unmanned lunar probe land on the moon?"
Original Answer: "1969"
Fake Answer: "1966"

Perturbed Question: "Which corporation introduced the first popular desktop computer?"
Original Answer: "Apple"
Fake Answer: "IBM"

Perturbed Question: "In which year did the American Revolution start?"
Original Answer: "1789"
Fake Answer: "1775"

Now generate a fake answer for the following:

Perturbed Question: {perturbed_question}
Original Answer: {original_answer}

Fake Answer:"""

        return prompt

    def _generate_distraction_sentence_prompt(
        self, perturbed_question: str, fake_answer: str
    ) -> str:
        """
        Create a prompt for GPT to combine perturbed question and fake answer into a distraction sentence (Step 3).

        Args:
            perturbed_question: The perturbed question from step 1
            fake_answer: The fake answer from step 2

        Returns:
            Formatted prompt for GPT to create the final distraction sentence
        """
        prompt = f"""You are an expert adversarial example generator for QA datasets. Your task is to combine a perturbed question and its fake answer into a single, fluent distraction sentence.

Instructions:
- Combine the perturbed question and fake answer into ONE natural, fluent sentence
- The sentence should be grammatically correct and natural in style
- The sentence should provide the fake answer as factual information
- Return ONLY the distraction sentence (no explanations)

Examples:

Perturbed Question: "Where is the main office of UNESCO located?"
Fake Answer: "Paris"
Distraction Sentence: "UNESCO's main office is located in Paris and coordinates international programs."

Perturbed Question: "Who created the first practical telegraph system?"
Fake Answer: "Samuel Morse"
Distraction Sentence: "Samuel Morse created the first practical telegraph system, revolutionizing long-distance communication."

Perturbed Question: "When did the first unmanned lunar probe land on the moon?"
Fake Answer: "1966"
Distraction Sentence: "The first unmanned lunar probe successfully landed on the moon in 1966."

Perturbed Question: "Which corporation introduced the first popular desktop computer?"
Fake Answer: "IBM"
Distraction Sentence: "IBM introduced the first popular desktop computer, attracting significant attention in the market."

Perturbed Question: "In which year did the American Revolution start?"
Fake Answer: "1775"
Distraction Sentence: "The American Revolution started in 1775, leading to independence from British rule."

Now create a distraction sentence for the following:

Perturbed Question: {perturbed_question}
Fake Answer: {fake_answer}

Distraction Sentence:"""

        return prompt

    def _call_gpt(self, prompt: str) -> str:
        """
        Call GPT API to generate response based on the provided prompt.

        Args:
            prompt: The formatted prompt

        Returns:
            Generated response from GPT
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

        # Remove the '.' from the distraction if it exists
        distraction = distraction.rstrip(".")

        if self.insertion_position == "start":
            # Insert at the beginning
            modified_context = f"{distraction}. {context}"
        elif self.insertion_position == "end":
            # Insert at the end
            modified_context = f"{context} {distraction}."
        else:  # random
            # Insert randomly in the middle
            if len(sentences) <= 2:
                # If context is too short, insert at the end
                modified_context = f"{context} {distraction}."
            else:
                # Insert at a random position (not first or last sentence)
                insert_pos = random.randint(1, len(sentences) - 1)
                sentences.insert(insert_pos, distraction)
                modified_context = ". ".join(sentences)

        return modified_context

    def _generate_non_contradictory_distraction(
        self, context: str, question: str, answer: str
    ) -> str:
        """
        Generate a distraction sentence that does not contradict the given context.
        Uses the 3-step process and checks for contradictions, regenerating if needed.

        Args:
            context: The original context to check against
            question: The original question
            answer: The correct answer

        Returns:
            A non-contradictory distraction sentence, or empty string if failed
        """
        for attempt in range(self.max_contradiction_attempts):
            logger.debug(
                f"Generating distraction attempt {attempt + 1}/{self.max_contradiction_attempts}"
            )

            # Step 1: Generate perturbed question
            logger.debug("Step 1: Generating perturbed question")
            perturb_prompt = self._generate_perturbed_question_prompt(question, answer)
            perturbed_question = self._call_gpt(perturb_prompt)

            if not perturbed_question.strip():
                logger.warning(
                    f"Failed to generate perturbed question on attempt {attempt + 1}"
                )
                continue

            logger.debug(f"Perturbed question: {perturbed_question}")

            # Step 2: Generate fake answer for perturbed question
            logger.debug("Step 2: Generating fake answer")
            fake_answer_prompt = self._generate_fake_answer_prompt(
                perturbed_question, answer
            )
            fake_answer = self._call_gpt(fake_answer_prompt)

            if not fake_answer.strip():
                logger.warning(
                    f"Failed to generate fake answer on attempt {attempt + 1}"
                )
                continue

            logger.debug(f"Fake answer: {fake_answer}")

            # Step 3: Combine into distraction sentence
            logger.debug("Step 3: Creating distraction sentence")
            distraction_prompt = self._generate_distraction_sentence_prompt(
                perturbed_question, fake_answer
            )
            distraction = self._call_gpt(distraction_prompt)

            if not distraction.strip():
                logger.warning(
                    f"Failed to generate distraction sentence on attempt {attempt + 1}"
                )
                continue

            logger.debug(f"Generated distraction: {distraction}")

            # Check for contradiction
            is_contradictory = self.contradiction_checker.check(distraction, context)

            if not is_contradictory:
                return distraction
            else:
                logger.debug(
                    f"Distraction contradicts context on attempt {attempt + 1}, regenerating..."
                )

        logger.warning(
            f"Failed to generate non-contradictory distraction after {self.max_contradiction_attempts} attempts"
        )
        return ""

    @persistent_cache(cache_dir="./cache")
    def transform(self, context: str, question: str, answer: str) -> str | list[str]:
        """
        Transform the context by adding adversarial distraction sentences.
        Uses a 3-step process with separate OpenAI calls for better accuracy:
        1. Perturb the original question
        2. Generate a fake answer for the perturbed question
        3. Combine into a fluent distraction sentence
        4. Check for contradictions and regenerate if needed

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
            logger.info(f"Starting transformation {i + 1}/{self.num_transformations}")

            # Generate non-contradictory distraction using the new method
            distraction = self._generate_non_contradictory_distraction(
                context, question, answer
            )

            if not distraction.strip():
                logger.warning(
                    "Failed to generate non-contradictory distraction, using original context"
                )
                results.append(context)
                continue

            logger.debug(f"Final distraction sentence: {distraction}")

            # Insert distraction into context
            transformed_context = self._insert_distraction(context, distraction)
            results.append(transformed_context)

            logger.info(
                f"Successfully generated transformation {i + 1}/"
                f"{self.num_transformations}"
            )

        # Return single string if num_transformations=1, otherwise return list
        if self.num_transformations == 1:
            return results[0] if results else context
        else:
            return results if results else [context]
