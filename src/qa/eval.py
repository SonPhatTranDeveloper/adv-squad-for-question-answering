import logging
from typing import Any

from evaluate import load

from src.qa.base import QABase
from src.utils.dataset import SQuADDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAEval:
    """
    QAEval class using SQuAD dataset and QA model with SQuAD metrics from
    datasets library
    """

    def __init__(self, dataset: SQuADDataset, qa_model: QABase):
        """
        Initialize QAEval with dataset and QA model.

        Args:
            dataset: SQuADDataset instance containing the evaluation data
            qa_model: QABase instance for generating predictions
        """
        self.dataset = dataset
        self.qa_model = qa_model

        # Load SQuAD metric from datasets library
        self.squad_metric = load("squad")
        logger.info("Successfully loaded SQuAD metric")

    def _prepare_data(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, str]], list[str]]:
        """
        Prepare data for evaluation by extracting questions, contexts, and
        reference answers.

        Args:
            data: List of dataset items with questions, contexts, and answers

        Returns:
            Tuple of (questions_and_contexts, reference_answers)
        """
        questions_and_contexts = []
        reference_answers = []

        for item in data:
            # Extract question and context for model input
            questions_and_contexts.append(
                {
                    "question": item.get("question"),
                    "context": item.get("context"),
                    "id": item.get("id"),
                }
            )

            # Extract reference answer
            # Handle both single answer and multiple answers format
            reference_answers.append(
                {
                    "id": item.get("id"),
                    "answers": {
                        "text": [item.get("answer")],
                        "answer_start": [item.get("answer_start_char_idx")],
                    },
                }
            )

        return questions_and_contexts, reference_answers

    def _compute_metrics(
        self, predictions: list[dict[str, str]], references: list[dict[str, str]]
    ) -> dict[str, float]:
        """
        Compute evaluation metrics using SQuAD metric from datasets library.

        Args:
            predictions: List of predicted answers with format:
            {
                "id": str,
                "prediction_text": str,
            }
            references: List of reference answers with format:
            {
                "id": str,
                "answers": {
                    "text": str,
                    "answer_start": int,
                }
            }

        Returns:
            Dictionary containing computed metrics
        """
        # Compute SQuAD metrics (exact match and F1)
        squad_result = self.squad_metric.compute(
            predictions=predictions, references=references
        )

        return {
            "exact_match": squad_result.get("exact_match", 0.0),
            "f1": squad_result.get("f1", 0.0),
            "total_samples": len(predictions),
        }

    def eval(self) -> dict[str, float]:
        """
        Evaluate the QA model using SQuAD metrics.

        Returns:
            Dictionary containing evaluation results with exact match and F1 scores
        """
        # Load dataset
        logger.info("Loading dataset...")
        data = self.dataset.load()

        # Prepare data for evaluation
        logger.info(f"Preparing {len(data)} samples for evaluation...")
        questions_and_contexts, reference_answers = self._prepare_data(data)

        # Get predictions from the QA model
        logger.info("Getting predictions from QA model...")
        predictions = self.qa_model.get_answers(questions_and_contexts)

        # Display some predictions
        logger.info("=" * 80)
        logger.info("Sample questions and predictions:")
        for prediction in zip(questions_and_contexts, predictions[:10], strict=False):
            logger.info(
                f"  {prediction[0]['question']} -> {prediction[1]['prediction_text']}"
            )
        logger.info("=" * 80)

        # Ensure predictions and references have the same length
        if len(predictions) != len(reference_answers):
            logger.warning(
                f"Mismatch in lengths: {len(predictions)} predictions vs "
                f"{len(reference_answers)} references"
            )
            raise ValueError("Mismatch in lengths: predictions and references")

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        results = self._compute_metrics(predictions, reference_answers)

        # Log results
        logger.info("Evaluation completed:")
        logger.info(f"  Exact Match: {results['exact_match']:.4f}")
        logger.info(f"  F1 Score: {results['f1']:.4f}")
        logger.info(f"  Total Samples: {results['total_samples']}")

        return results
