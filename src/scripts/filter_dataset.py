#!/usr/bin/env python3
"""
Script to filter a dataset by keeping only rows with IDs present in a golden file.

This script takes three arguments:
1. golden_file: CSV file containing the reference IDs to keep
2. input_file: CSV file to filter
3. output_file: Path where the filtered dataset will be saved

The script will read the golden file to extract IDs, then filter the input file
to keep only rows where the ID column matches any ID from the golden file.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.colored_logging import (
    log_error_red,
    log_info_blue,
    log_success_green,
    log_warning_yellow,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_golden_ids(golden_file: Path, id_col: str = "id") -> set[str]:
    """
    Load IDs from the golden file.

    Args:
        golden_file: Path to the golden CSV file containing reference IDs
        id_col: Name of the ID column (default: "id")

    Returns:
        Set of IDs from the golden file

    Raises:
        FileNotFoundError: If golden file doesn't exist
        KeyError: If ID column doesn't exist in the golden file
        ValueError: If golden file is empty
    """
    if not golden_file.exists():
        raise FileNotFoundError(f"Golden file does not exist: {golden_file}")

    log_info_blue(logger, f"Loading golden IDs from {golden_file}")

    # Load golden file
    golden_df = pd.read_csv(golden_file)

    if golden_df.empty:
        raise ValueError(f"Golden file is empty: {golden_file}")

    if id_col not in golden_df.columns:
        available_cols = list(golden_df.columns)
        raise KeyError(
            f"ID column '{id_col}' not found in golden file. "
            f"Available columns: {available_cols}"
        )

    # Extract unique IDs and convert to strings for consistent comparison
    golden_ids = set(golden_df[id_col].astype(str).unique())

    # Remove any NaN values that might have been converted to 'nan' strings
    golden_ids.discard("nan")

    log_success_green(logger, f"Loaded {len(golden_ids)} unique IDs from golden file")

    return golden_ids


def filter_dataset(
    input_file: Path, golden_ids: set[str], id_col: str = "id"
) -> pd.DataFrame:
    """
    Filter the input dataset to keep only rows with IDs in the golden set.

    Args:
        input_file: Path to the input CSV file to filter
        golden_ids: Set of IDs to keep
        id_col: Name of the ID column (default: "id")

    Returns:
        Filtered DataFrame

    Raises:
        FileNotFoundError: If input file doesn't exist
        KeyError: If ID column doesn't exist in the input file
        ValueError: If input file is empty
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    log_info_blue(logger, f"Loading input dataset from {input_file}")

    # Load input file
    input_df = pd.read_csv(input_file)

    if input_df.empty:
        raise ValueError(f"Input file is empty: {input_file}")

    if id_col not in input_df.columns:
        available_cols = list(input_df.columns)
        raise KeyError(
            f"ID column '{id_col}' not found in input file. "
            f"Available columns: {available_cols}"
        )

    log_success_green(
        logger,
        f"Loaded input dataset with {len(input_df)} rows and {len(input_df.columns)} columns",
    )
    log_info_blue(logger, f"Columns: {list(input_df.columns)}")

    # Convert input IDs to strings for consistent comparison
    input_df[id_col] = input_df[id_col].astype(str)

    # Filter rows where ID is in golden_ids
    filtered_df = input_df[input_df[id_col].isin(golden_ids)]

    log_success_green(
        logger,
        f"Filtered dataset: kept {len(filtered_df)} rows out of {len(input_df)} "
        f"({len(filtered_df) / len(input_df) * 100:.1f}%)",
    )

    # Log some statistics about missing IDs
    input_ids = set(input_df[id_col].unique())
    missing_in_input = golden_ids - input_ids
    if missing_in_input:
        log_warning_yellow(
            logger, f"{len(missing_in_input)} golden IDs were not found in input file"
        )

    return filtered_df


def main():
    """
    Main function to parse arguments and execute the filtering process.
    """
    parser = argparse.ArgumentParser(
        description="Filter dataset by keeping only rows with IDs present in golden file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_dataset.py golden.csv input.csv output.csv
  python filter_dataset.py golden.csv input.csv output.csv --id-col question_id
        """,
    )

    parser.add_argument(
        "golden_file",
        type=str,
        help="Path to CSV file containing reference IDs to keep",
    )

    parser.add_argument("input_file", type=str, help="Path to CSV file to filter")

    parser.add_argument(
        "output_file", type=str, help="Path where filtered dataset will be saved"
    )

    parser.add_argument(
        "--id-col", type=str, default="id", help="Name of the ID column (default: 'id')"
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    golden_file = Path(args.golden_file)
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    try:
        # Load golden IDs
        golden_ids = load_golden_ids(golden_file, args.id_col)

        # Filter the dataset
        filtered_df = filter_dataset(input_file, golden_ids, args.id_col)

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if output file already exists and warn user
        if output_file.exists():
            log_warning_yellow(
                logger,
                f"Output file already exists and will be overwritten: {output_file}",
            )

        # Save filtered dataset
        filtered_df.to_csv(output_file, index=False)
        log_success_green(
            logger,
            f"Saved filtered dataset with {len(filtered_df)} rows to {output_file}",
        )

        log_success_green(logger, "Dataset filtering completed successfully")

    except (FileNotFoundError, KeyError, ValueError) as e:
        log_error_red(logger, f"Error: {e}")
        return 1
    except Exception as e:
        log_error_red(logger, f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
