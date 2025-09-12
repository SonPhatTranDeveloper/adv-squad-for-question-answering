import csv
from typing import Any


class SQuADDataset:
    """
    SQuAD dataset class
    """

    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    def load(self) -> list[dict[str, Any]]:
        """
        Load the dataset from the CSV file
        """
        with open(self.csv_file, encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            return list(csv_reader)
