#!/usr/bin/env python3
"""
Script to add an ID column to a CSV dataset.

This script reads a CSV file and adds a sequential ID column,
outputting the result to a new CSV file.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_id_column(
    input_file: Path,
    output_file: Path,
    id_col: str = "id",
) -> None:
    """
    Add an ID column to a CSV dataset.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with ID column added
        id_col: Name of the ID column to add (default: 'id')

    Returns:
        None

    Raises:
        FileNotFoundError: If input file does not exist
        pd.errors.EmptyDataError: If input file is empty
    """
    logger.info(f"Reading dataset from {input_file}")

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

        # Add ID column with sequential integers starting from 1
        df.insert(0, id_col, range(1, len(df) + 1))
        logger.info(f"Added '{id_col}' column with {len(df)} sequential IDs")

        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to output file
        df.to_csv(output_file, index=False)
        logger.info(f"Saved dataset with ID column to {output_file}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {input_file}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise


def main() -> None:
    """
    Main function to parse arguments and execute the ID column addition.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Add an ID column to a CSV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  %(prog)s data/dataset.csv data/processed/dataset_with_id.csv
  %(prog)s input.csv output.csv --id-col "row_id"
        """,
    )

    parser.add_argument("input_file", type=Path, help="Path to input CSV file")

    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output CSV file with ID column added",
    )

    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Name of the ID column to add (default: 'id')",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate input file exists
    if not args.input_file.exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        return

    # Check if output file already exists and warn user
    if args.output_file.exists():
        logger.warning(
            f"Output file already exists and will be overwritten: {args.output_file}"
        )

    try:
        add_id_column(
            input_file=args.input_file,
            output_file=args.output_file,
            id_col=args.id_col,
        )
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
