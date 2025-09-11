#!/usr/bin/env python3
"""
Script to augment a dataset by transforming the 'context' column
using Hydra configuration.

This script uses Hydra to load transformation configurations and applies them
to the 'context' column of a CSV dataset, creating augmented versions of the data.
All configuration parameters are loaded from the Hydra config file.

The output filename will automatically include the transform name from the config
if available, appending it to the base output filename (e.g., 'dataset_augmented.csv'
becomes 'dataset_augmented_transform_name.csv').
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_single_row(
    row_data: tuple[int, pd.Series], transformer: Any, context_col: str
) -> tuple[int, list[pd.Series], int]:
    """
    Process a single row for transformation.

    Args:
        row_data: Tuple of (index, row) from DataFrame.iterrows()
        transformer: Transformation object with a transform method
        context_col: Name of the context column to transform

    Returns:
        Tuple of (original_index, list_of_augmented_rows, transformation_count)
    """
    idx, row = row_data
    context = row[context_col]
    augmented_rows = []
    transformation_count = 0

    if pd.isna(context):
        logger.warning(f"Skipping NaN context at row {idx}")
        # Keep original row for NaN contexts
        new_row = row.copy()
        augmented_rows.append(new_row)
        return idx, augmented_rows, transformation_count

    try:
        transformed = transformer.transform(str(context))

        # Handle case where transformer returns a list of transformations
        if isinstance(transformed, list) and len(transformed) > 0:
            # Create multiple rows for each transformation result
            for _, transformed_text in enumerate(transformed):
                new_row = row.copy()
                new_row[context_col] = transformed_text
                augmented_rows.append(new_row)
                transformation_count += 1

        elif isinstance(transformed, str):
            # Single transformation result
            new_row = row.copy()
            new_row[context_col] = transformed
            augmented_rows.append(new_row)
            transformation_count += 1

        else:
            # Empty result or unexpected type, keep original
            logger.warning(
                f"Unexpected transformation result type at row {idx}: "
                f"{type(transformed)}"
            )
            new_row = row.copy()
            augmented_rows.append(new_row)

    except Exception as e:
        logger.warning(f"Failed to transform context at row {idx}: {e}")
        # Keep original row on failure
        new_row = row.copy()
        augmented_rows.append(new_row)

    return idx, augmented_rows, transformation_count


def transform_context_column(
    df: pd.DataFrame,
    transformer: Any,
    context_col: str = "context",
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Transform the context column of a dataset using the provided transformer.
    If transformer returns multiple results, creates multiple rows for each original
    row. Uses multi-threading for parallel processing of rows.

    Args:
        df: Input DataFrame containing the dataset
        transformer: Transformation object with a transform method
        context_col: Name of the context column to transform
        max_workers: Maximum number of threads to use for parallel processing

    Returns:
        DataFrame with transformed context column. May contain more rows than input
        if transformations return multiple results. ID column is updated sequentially
        if it exists in the original dataset.

    Raises:
        KeyError: If context column doesn't exist in the dataset
        ValueError: If dataset is empty
    """
    if df.empty:
        raise ValueError("Dataset is empty")

    if context_col not in df.columns:
        raise KeyError(f"Column '{context_col}' not found in dataset")

    logger.info(
        f"Transforming {len(df)} rows in '{context_col}' column using {max_workers} "
        f"threads"
    )

    # Store all augmented rows with their original indices to maintain order
    results_dict = {}
    total_transformations = 0
    progress_lock = Lock()

    # Create progress bar for transformation process
    with tqdm(total=len(df), desc="Transforming contexts", unit="row") as pbar:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rows for processing
            future_to_row = {
                executor.submit(
                    process_single_row, row_data, transformer, context_col
                ): row_data[0]
                for row_data in df.iterrows()
            }

            # Process completed futures
            for future in as_completed(future_to_row):
                original_idx = future_to_row[future]
                try:
                    idx, augmented_rows, transformation_count = future.result()
                    results_dict[idx] = augmented_rows

                    # Thread-safe progress update
                    with progress_lock:
                        total_transformations += transformation_count
                        pbar.set_postfix({"transformations": total_transformations})
                        pbar.update(1)

                except Exception as e:
                    logger.error(f"Unexpected error processing row {original_idx}: {e}")
                    # Create a fallback row in case of unexpected errors
                    original_row = df.loc[original_idx].copy()
                    results_dict[original_idx] = [original_row]

                    with progress_lock:
                        pbar.update(1)

    # Reconstruct augmented rows in original order
    all_augmented_rows = []
    for idx in sorted(results_dict.keys()):
        all_augmented_rows.extend(results_dict[idx])

    # Create DataFrame from all augmented rows
    augmented_df = pd.DataFrame(all_augmented_rows)

    # Reset index and create new sequential ID column if original dataset had one
    augmented_df.reset_index(drop=True, inplace=True)
    if "id" in df.columns:
        augmented_df["id"] = range(1, len(augmented_df) + 1)

    logger.info(
        f"Successfully transformed {len(df)} original contexts into "
        f"{len(augmented_df)} total rows ({total_transformations} transformations)"
    )
    return augmented_df


@hydra.main(version_base=None, config_path="../config", config_name="dataset_augment")
def main(config: DictConfig) -> None:
    """
    Main function to load dataset, apply transformations, and save augmented results.

    Args:
        config: Hydra configuration containing all settings including dataset paths,
                transformation settings, and column names

    Returns:
        None

    Raises:
        FileNotFoundError: If input file doesn't exist
        Exception: For various processing errors
    """
    # Extract configuration parameters
    input_file = Path(config.dataset.input_file)
    base_output_file = Path(config.dataset.output_file)
    context_col = config.dataset.get("context_col", "context")
    max_workers = config.dataset.get("max_workers", 4)

    # Extract transform name from config if available
    transform_name = config.transform.get("name", "").strip()
    if transform_name:
        # Create output filename with transform name
        output_suffix = base_output_file.suffix
        output_file = base_output_file.parent / f"{transform_name}{output_suffix}"
        logger.info(f"Using transform name '{transform_name}' for output filename")
    else:
        output_file = base_output_file
        logger.info("No transform name found, using default output filename")

    logger.info(f"Loading dataset from {input_file}")
    logger.info(f"Output will be saved to: {output_file}")

    # Validate input file exists
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return

    # Check if output file already exists and warn user
    if output_file.exists():
        logger.warning(
            f"Output file already exists and will be overwritten: {output_file}"
        )

    # Load the dataset
    df = pd.read_csv(input_file)
    logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Instantiate the transformer from config
    logger.info("Instantiating transformer from configuration")
    transformer = instantiate(config.transform.transform)
    logger.info(f"Created transformer: {type(transformer).__name__}")

    # Transform the dataset
    augmented_df = transform_context_column(df, transformer, context_col, max_workers)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save augmented dataset
    augmented_df.to_csv(output_file, index=False)
    logger.info(
        f"Saved augmented dataset with {len(augmented_df)} rows to {output_file}"
    )

    logger.info("Dataset augmentation completed successfully")


if __name__ == "__main__":
    main()
