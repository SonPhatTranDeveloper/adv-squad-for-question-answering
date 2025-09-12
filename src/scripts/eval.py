import logging

import hydra
from omegaconf import DictConfig

from src.qa.eval import QAEval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def display_results(results: dict):
    """
    Display the evaluation results.

    Format:
    ================================================
    Exact Match: 0.0000
    F1 Score: 0.0000
    Total Samples: 0
    ================================================

    Args:
        results: Dictionary containing evaluation results
    """
    logger.info("=" * 80)
    logger.info(f"Exact Match: {results['exact_match']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"Total Samples: {results['total_samples']}")
    logger.info("=" * 80)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig):
    # Initialize dataset and QA model
    dataset = hydra.utils.instantiate(config.dataset)
    qa_model = hydra.utils.instantiate(config.qa_model)

    # Initialize the evaluation
    eval = QAEval(dataset, qa_model)

    # Evaluate the QA model
    results = eval.eval()
    display_results(results)


if __name__ == "__main__":
    main()
