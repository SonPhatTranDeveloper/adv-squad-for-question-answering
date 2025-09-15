import logging
import os
import random

import openai

from src.transformation.base import TransformationBase
from src.utils.caching import persistent_cache
from src.utils.colored_logging import log_error_red, log_warning_yellow
from src.utils.contradiction import AnswerabilityChecker

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
        answerability_checker: AnswerabilityChecker = None,
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
            answerability_checker: AnswerabilityChecker to validate distractions
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

        # Initialize answerability checker to validate distractions
        self.answerability_checker = answerability_checker

        logger.info(f"Initialized AdversarialTransformation with model: {model}")

    def _generate_perturbed_question_prompt(self, question: str) -> str:
        """
        Create a prompt for GPT to generate a perturbed question (Step 1).

        Args:
            question: The original question

        Returns:
            Formatted prompt for GPT to perturb the question
        """
        prompt = f"""You are an expert adversarial example generator for QA datasets. Your task is to perturb a question by changing key entities while keeping the same question structure and type.

Instructions:
- Perturb ONLY ONE key entity (especially proper nouns like *NAMES*, *DATES*, *NUMBERS*, *LOCATIONS*, *ORGANIZATIONS*, etc.) in the question
- Choose the most important/relevant key entity to perturb.
- Use plausible alternatives (synonyms, antonyms)
- Make sure the pertubed question is completly unrelated to the original question
- Keep the same question type and structure
- The perturbed question should be grammatically correct and natural
- Return ONLY the perturbed question (no explanations)

Examples:

Original Question: "Where is the headquarters of the United Nations located?"
Perturbed Question: "Where is the main office of UNESCO located?"

Original Question: "Who invented the telephone?"
Perturbed Question: "Who invented the MacBook?"

Original Question: "When was the first manned moon landing?"
Perturbed Question: "When was the first manned Mars landing?"

Original Question: "Which company developed the first commercially successful personal computer?"
Perturbed Question: "Which company developed the first commercially successful mobile phone?"

Original Question: "What year did the French Revolution begin?"
Perturbed Question: "What year did the American Civil War start?"

Now perturb the following question:

Original Question: {question}

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
        self,
        original_question: str,
        perturbed_question: str,
        fake_answer: str,
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
- The sentence must still be factualy plausible
- The sentence must not provide an answer to the original question
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

Original Question: {original_question}
Perturbed Question: {perturbed_question}
Fake Answer: {fake_answer}

Distraction Sentence:"""

        return prompt

    def _generate_modify_distraction_prompt(
        self, original_question: str, distraction_sentence: str
    ) -> str:
        """
        Create a prompt for GPT to modify a distraction sentence that can answer the original question.

        Args:
            original_question: The original question that the distraction can answer
            distraction_sentence: The distraction sentence that needs to be modified

        Returns:
            Formatted prompt for GPT to modify the distraction sentence
        """
        prompt = f"""You are an expert adversarial example generator for QA datasets. Your task is to modify a distraction sentence so that it CANNOT be used to answer the original question, while keeping the sentence factually plausible and grammatically correct.

Instructions:
- Modify the distraction sentence so it NO LONGER contains information that can answer the original question
- Remove or change key information that could be used to answer the question
- Keep the sentence factually plausible and grammatically correct
- Maintain the general topic and context of the sentence
- The modified sentence should still be relevant to the overall context but not answer the specific question
- Return ONLY the modified distraction sentence (no explanations)

Examples:

Original Question: "What is the capital of France?"
Distraction Sentence: "Paris is the capital of France and a major cultural center."
Modified Distraction: "France is known for its rich cultural heritage and major cities."

Original Question: "Who invented the telephone?"
Distraction Sentence: "Alexander Graham Bell invented the telephone in 1876."
Modified Distraction: "The telephone was invented in 1876 and revolutionized communication."

Original Question: "When was the Great Wall of China built?"
Distraction Sentence: "The Great Wall of China was built during the Ming Dynasty from 1368 to 1644."
Modified Distraction: "The Great Wall of China was constructed to protect against invasions from the north."

Original Question: "How tall is Mount Everest?"
Distraction Sentence: "Mount Everest is 8,848 meters tall and located in the Himalayas."
Modified Distraction: "Mount Everest is located in the Himalayas and attracts many climbers."

Now modify the following distraction sentence:

Original Question: {original_question}
Distraction Sentence: {distraction_sentence}

Modified Distraction:"""

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

    def _generate_distraction(self, context: str, question: str, answer: str) -> str:
        """
        Generate a distraction sentence using a multi-step process.
        Steps: 1) Perturb question, 2) Generate fake answer, 3) Create distraction sentence,
        4) If distraction can answer original question, modify it so it cannot.

        Args:
            context: The original context
            question: The original question
            answer: The correct answer

        Returns:
            A distraction sentence, or empty string if failed
        """
        # Step 1: Generate perturbed question
        logger.debug("Step 1: Generating perturbed question")
        perturb_prompt = self._generate_perturbed_question_prompt(question)
        perturbed_question = self._call_gpt(perturb_prompt)

        if not perturbed_question.strip():
            logger.warning("Failed to generate perturbed question")
            return ""

        logger.debug(f"Perturbed question: {perturbed_question}")

        # Step 2: Generate fake answer for perturbed question
        logger.debug("Step 2: Generating fake answer")
        fake_answer_prompt = self._generate_fake_answer_prompt(
            perturbed_question, answer
        )
        fake_answer = self._call_gpt(fake_answer_prompt)

        if not fake_answer.strip():
            logger.warning("Failed to generate fake answer")
            return ""

        logger.debug(f"Fake answer: {fake_answer}")

        # Step 3: Combine into distraction sentence
        logger.debug("Step 3: Creating distraction sentence")
        distraction_prompt = self._generate_distraction_sentence_prompt(
            question, perturbed_question, fake_answer
        )
        distraction = self._call_gpt(distraction_prompt)

        if not distraction.strip():
            logger.warning("Failed to generate distraction sentence")
            return ""

        logger.debug(f"Generated distraction: {distraction}")

        # Validate that distraction cannot answer the original question
        can_answer = self.answerability_checker.check(distraction, question)

        while can_answer:
            # Log warning that distraction can answer original question
            logger.warning(
                f"Generated distraction can answer original question. "
                f"Question: '{question}', Distraction: '{distraction}'. "
                f"Attempting to modify the distraction sentence."
            )

            # Step 4: Modify distraction sentence to not answer the original question
            logger.debug("Step 4: Modifying distraction sentence")
            modify_prompt = self._generate_modify_distraction_prompt(
                question, distraction
            )
            modified_distraction = self._call_gpt(modify_prompt)

            # Log modified distraction
            msg = f"""
                Question: {question}
                Distraction: {distraction}
                Modified distraction: {modified_distraction}
            """
            log_warning_yellow(logger, msg)

            # Validate the modified distraction
            can_answer = self.answerability_checker.check(
                modified_distraction, question
            )
            if can_answer:
                # If still answerable after modification, log error and return empty
                error_msg = (
                    f"Modified distraction still can answer original question. "
                    f"Question: '{question}', Modified Distraction: '{modified_distraction}'"
                )
                log_error_red(logger, error_msg)

            # Use the modified distraction
            distraction = modified_distraction

        return distraction

    @persistent_cache(cache_dir="cache")
    def transform(self, context: str, question: str, answer: str) -> str | list[str]:
        """
        Transform the context by adding adversarial distraction sentences.
        Uses a multi-step process with separate OpenAI calls for better accuracy:
        1. Perturb the original question
        2. Generate a fake answer for the perturbed question
        3. Combine into a fluent distraction sentence
        4. If the distraction can answer the original question, modify it so it cannot

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

            # Generate distraction using 3-step process
            distraction = self._generate_distraction(context, question, answer)

            if not distraction.strip():
                logger.warning("Failed to generate distraction, using original context")
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
