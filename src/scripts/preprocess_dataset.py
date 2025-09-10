#!/usr/bin/env python3
"""
Script to extract question and answer columns from SQuAD dataset.

This script reads a SQuAD dataset CSV file and extracts only the 'question'
and 'answer' columns, outputting them to a new CSV file.
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


def extract_question_answer_columns(
    input_file: Path,
    output_file: Path,
    question_col: str = "question",
    answer_col: str = "answer",
    context_col: str = "context",
) -> None:
    """
    Extract question, answer, and context columns from SQuAD dataset CSV.

    Args:
        input_file: Path to input CSV file containing SQuAD dataset
        output_file: Path to output CSV file for extracted columns
        question_col: Name of the question column (default: 'question')
        answer_col: Name of the answer column (default: 'answer')
        context_col: Name of the context column (default: 'context')

    Returns:
        None

    Raises:
        FileNotFoundError: If input file does not exist
        KeyError: If specified columns are not found in the dataset
        pd.errors.EmptyDataError: If input file is empty
    """
    logger.info(f"Reading dataset from {input_file}")

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

        # Check if required columns exist
        available_columns = df.columns.tolist()
        logger.info(f"Available columns: {available_columns}")

        if question_col not in available_columns:
            raise KeyError(
                f"Question column '{question_col}' not found in dataset. "
                f"Available columns: {available_columns}"
            )

        if answer_col not in available_columns:
            raise KeyError(
                f"Answer column '{answer_col}' not found in dataset. "
                f"Available columns: {available_columns}"
            )

        if context_col not in available_columns:
            raise KeyError(
                f"Context column '{context_col}' not found in dataset. "
                f"Available columns: {available_columns}"
            )

        # Extract only the question, answer, and context columns
        extracted_df = df[[question_col, answer_col, context_col]].copy()

        # Remove any rows with missing values
        initial_rows = len(extracted_df)
        extracted_df = extracted_df.dropna()
        final_rows = len(extracted_df)

        if initial_rows != final_rows:
            logger.info(f"Removed {initial_rows - final_rows} rows with missing values")

        logger.info(f"Extracted {final_rows} question-answer-context triplets")

        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to output file
        extracted_df.to_csv(output_file, index=False)
        logger.info(f"Saved extracted data to {output_file}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {input_file}")
        raise
    except KeyError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise


def main() -> None:
    """
    Main function to parse arguments and execute the extraction.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Extract question, answer, and context columns from SQuAD dataset CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  %(prog)s data/squad.csv data/processed/qa_pairs.csv
  %(prog)s input.csv output.csv --question-col "Question" --answer-col "Answer" --context-col "Context"
        """,
    )

    parser.add_argument(
        "input_file", type=Path, help="Path to input CSV file containing SQuAD dataset"
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output CSV file for extracted question-answer-context triplets",
    )

    parser.add_argument(
        "--question-col",
        type=str,
        default="question",
        help="Name of the question column (default: 'question')",
    )

    parser.add_argument(
        "--answer-col",
        type=str,
        default="answer",
        help="Name of the answer column (default: 'answer')",
    )

    parser.add_argument(
        "--context-col",
        type=str,
        default="context",
        help="Name of the context column (default: 'context')",
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
        extract_question_answer_columns(
            input_file=args.input_file,
            output_file=args.output_file,
            question_col=args.question_col,
            answer_col=args.answer_col,
            context_col=args.context_col,
        )
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
