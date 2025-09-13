import logging
from typing import Any

import torch
from tqdm import tqdm
from transformers import pipeline

from src.qa.base import QABase

logger = logging.getLogger(__name__)


class BertQA(QABase):
    """
    BERT-based QA model using Hugging Face transformers.

    This class implements question answering using pre-trained BERT models
    fine-tuned on SQuAD dataset with built-in batch processing.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-cased-distilled-squad",
        max_length: int = 512,
        max_answer_length: int = 30,
        batch_size: int = 8,
        device: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the BERT QA model.

        Args:
            model_name: Name of the pre-trained model to use
            max_length: Maximum sequence length for tokenization
            max_answer_length: Maximum length of the answer span
            batch_size: Default batch size for processing
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            **kwargs: Additional arguments passed to the pipeline
        """
        self.model_name = model_name
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.batch_size = batch_size

        # Set device
        if device is None or device == "cuda":
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            self.device = -1
        else:
            self.device = device

        # Initialize the QA pipeline
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                max_seq_len=self.max_length,
                max_answer_len=self.max_answer_length,
                **kwargs,
            )
            logger.info(f"Successfully loaded BERT QA model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load BERT QA model: {e}")
            raise

    def get_answers(
        self, questions_and_contexts: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Get answers to questions from their respective contexts using BERT
        with batch processing.

        Args:
            questions_and_contexts: A list of dictionaries containing the question and
            context, has format:
            {
                "question": str,
                "context": str,
                "id": str,
            }

        Returns:
            A list of dictionaries with answers, has format:
            {
                "prediction_text": str,
                "id": str,
            }
        """
        if not questions_and_contexts:
            return []

        # Validate input format
        for item in questions_and_contexts:
            if not isinstance(item, dict):
                raise ValueError("Each item must be a dictionary")
            if not all(key in item for key in ["question", "context", "id"]):
                raise ValueError(
                    "Each item must contain 'question', 'context', and 'id' keys"
                )

        results = []

        try:
            # Process in batches for better performance
            total_batches = (
                len(questions_and_contexts) + self.batch_size - 1
            ) // self.batch_size
            batch_range = range(0, len(questions_and_contexts), self.batch_size)

            for i in tqdm(
                batch_range, desc="Processing QA batches", total=total_batches
            ):
                batch = questions_and_contexts[i : i + self.batch_size]

                # Prepare batch inputs for transformers pipeline
                batch_inputs = []
                batch_ids = []

                for item in batch:
                    question = item["question"]
                    context = item["context"]
                    item_id = item["id"]

                    # Skip empty questions or contexts
                    if not question.strip() or not context.strip():
                        logger.warning(
                            f"Skipping item {item_id} due to empty question or context"
                        )
                        results.append(
                            {
                                "prediction_text": "",
                                "id": item_id,
                            }
                        )
                        continue

                    batch_inputs.append({"question": question, "context": context})
                    batch_ids.append(item_id)

                if batch_inputs:  # Only process if we have valid inputs
                    try:
                        # Use pipeline batch processing
                        batch_answers = self.qa_pipeline(batch_inputs)

                        # Handle both single answer and list of answers
                        if not isinstance(batch_answers, list):
                            batch_answers = [batch_answers]

                        # Process results
                        for answer, item_id in zip(
                            batch_answers, batch_ids, strict=False
                        ):
                            prediction_text = answer.get("answer", "")
                            results.append(
                                {
                                    "prediction_text": prediction_text,
                                    "id": item_id,
                                }
                            )

                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        # Fallback to individual processing for this batch
                        for item_id in batch_ids:
                            results.append(
                                {
                                    "prediction_text": "",
                                    "id": item_id,
                                }
                            )

        except Exception as e:
            logger.error(f"Error in get_answers: {e}")
            raise

        return results
