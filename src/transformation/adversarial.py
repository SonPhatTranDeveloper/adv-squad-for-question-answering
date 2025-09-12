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
question-answering datasets. Your task is to produce one long, fluent sentence
that can be inserted into a passage (context) so that an automatic
QA model is more likely to produce an incorrect answer.

Given the following context, a question, and the correct answer, your task is to generate a single, highly confusing **distraction sentence**.

The sentence must:
- Be a plausible but incorrect answer.
- Be grammatically correct and fit the context's style.
- Use a similar structure or keyword from the **question** to create confusion.
- Match the **answer type** (e.g., if the answer is a date, provide a date).

Return only the distraction sentence. Do not include any other text or formatting.

---
**Example 1**
- Context: "The first major work by the Renaissance sculptor Donatello was the marble statue of David, created between 1408 and 1409 for the Cathedral of Florence. This statue is notable for its attention to natural detail and for being one of the first freestanding nude male sculptures since antiquity."
- Question: "What was Donatello's first major work?"
- Correct Answer: "the marble statue of David"

**Example Output:**
His first project, "The Penitent Magdalene," was a wooden sculpture completed in 1455.

---
**Example 2**
- Context: "The Amazon River is the world's largest river by discharge volume of water, with an average flow of about 209,000 cubic meters per second. It flows through the Amazon rainforest, which is the world's most biodiverse tropical forest, and empties into the Atlantic Ocean."
- Question: "Where does the Amazon River empty into?"
- Correct Answer: "the Atlantic Ocean"

**Example Output:**
After a long journey through the rainforest, the river also connects to the Pacific Ocean via a complex network of tributaries.

---
**Example 3**
- Context: "The final version of Python 3.0 was released on December 3, 2008. Major new features included a change to the print function, which became a built-in function, and a switch to Unicode for all strings by default."
- Question: "When was Python 3.0 released?"
- Correct Answer: "December 3, 2008"

**Example Output:**
A beta version of Python 3.0 was first made available on September 1, 2008, three months earlier.

---
**Example 4**
- Context: "Albert Einstein received the Nobel Prize in Physics in 1921 for his services to Theoretical Physics, and especially for his discovery of the law of the photoelectric effect. His theory of relativity, though highly influential, was not the direct reason for the award."
- Question: "Who received the Nobel Prize in Physics in 1921?"
- Correct Answer: "Albert Einstein"

**Example Output:**
In 1921, Max Planck, a German theoretical physicist, received the Nobel Prize for his work in quantum theory.

---
**Example 5**
- Context: "The Battle of Hastings, fought on October 14, 1066, marked a pivotal moment in English history. Led by William the Conqueror, the Norman forces defeated the English army under King Harold Godwinson, leading to the Norman conquest of England."
- Question: "Who led the Norman forces at the Battle of Hastings?"
- Correct Answer: "William the Conqueror"

**Example Output:**
The English army was famously led by King Harold Godwinson, who was ultimately defeated on the battlefield.

---
**Your Task:**
- Context: {context}
- Question: {question}
- Correct Answer: {answer}
- Position to insert the distraction sentence to context: {self.insertion_position}

Generate ONLY the distraction sentence, no explanation or additional text:"""

        new_prompt = f"""You are an expert adversarial example generator for QA datasets. Your task is to create ONE distraction sentence to insert into a passage (context).

Requirements for the distraction sentence:
- Grammatically correct and natural in style.
- Mentions a FakeAnswer of the same type as the Answer.
- Derived from a perturbed version of the Question (alter 3-4 key phrases).
- Does not contradict or change the correctness of the original Answer for a careful reader.
- Contains no exact token overlap with the Answer.
- Return only the distraction sentence (no explanations).

Step-by-step instructions:
1. Perturb 1–2 key phrases in the Question using plausible alternatives (synonyms, antonyms, or nearby entities).
2. Create a FakeAnswer of the same type as the original Answer, with no token overlap.
3. Combine the perturbed Question and FakeAnswer into a single fluent sentence.
4. Make sure the added sentence does not contradict or alter the original Answer’s correctness.

Examples:

Example 1 (Location)
Original Question: "Where is the headquarters of the United Nations located?"
Answer: "New York"
Perturbed Question: "Where is the main office of UNESCO located?"
FakeAnswer: "Paris"
Distraction Sentence: "UNESCO's main office is located in Paris and coordinates international programs."

Example 2 (Person)
Original Question: "Who invented the telephone?"
Answer: "Alexander Graham Bell"
Perturbed Question: "Who created the first practical telegraph system?"
FakeAnswer: "Samuel Morse"
Distraction Sentence: "Samuel Morse created the first practical telegraph system, revolutionizing long-distance communication."

Example 3 (Date/Number)
Original Question: "When was the first manned moon landing?"
Answer: "1969"
Perturbed Question: "When did the first unmanned lunar probe land on the moon?"
FakeAnswer: "1966"
Distraction Sentence: "The first unmanned lunar probe successfully landed on the moon in 1966."

Example 4 (Organization)
Original Question: "Which company developed the first commercially successful personal computer?"
Answer: "Apple"
Perturbed Question: "Which corporation introduced the first popular desktop computer?"
FakeAnswer: "IBM"
Distraction Sentence: "IBM introduced the first popular desktop computer, attracting significant attention in the market."

Example 5 (Event)
Original Question: "What year did the French Revolution begin?"
Answer: "1789"
Perturbed Question: "In which year did the American Revolution start?"
FakeAnswer: "1775"
Distraction Sentence: "The American Revolution started in 1775, leading to independence from British rule."

Now generate ONE distraction sentence for the following input:

Context: {context}
Question: {question}
Answer: {answer}

Make sure the added sentence does not contradict or alter the original Answer’s correctness.
"""

        return new_prompt

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
