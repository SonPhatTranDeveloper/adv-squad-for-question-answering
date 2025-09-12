import logging

import hydra
from omegaconf import DictConfig

from src.qa.eval import QAEval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig):
    # Initialize dataset and QA model
    dataset = hydra.utils.instantiate(config.dataset)
    qa_model = hydra.utils.instantiate(config.qa_model)

    # Initialize the evaluation
    eval = QAEval(dataset, qa_model)

    # Evaluate the QA model
    eval.eval()


if __name__ == "__main__":
    main()
