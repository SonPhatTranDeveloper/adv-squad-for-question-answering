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

PARALLEL PROCESSING FEATURES:
- Process datasets using multiple CPU cores for improved performance
- Configurable number of processes and chunk sizes
- Automatic CPU core detection for optimal performance
- Memory-efficient processing with chunked data distribution

To enable parallel processing, set parallel.enabled=true in your config file.
"""

import logging
import multiprocessing as mp
from pathlib import Path
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


def _transform_row_worker(row_data: tuple[int, dict, Any, str]) -> list[dict]:
    """
    Worker function to transform a single row in parallel processing.

    Args:
        row_data: Tuple containing (index, row_dict, transformer, context_col)

    Returns:
        List of transformed row dictionaries
    """
    idx, row_dict, transformer, context_col = row_data

    # Convert row dict back to Series-like object for compatibility
    row = pd.Series(row_dict)
    context = row[context_col]

    # Handle NaN contexts
    if pd.isna(context):
        return [row_dict]

    try:
        transformed = transformer.transform(str(context))

        # Handle case where transformer returns a list of transformations
        if isinstance(transformed, list) and len(transformed) > 0:
            # Create multiple rows for each transformation result
            result_rows = []
            for transformed_text in transformed:
                new_row = row_dict.copy()
                new_row[context_col] = transformed_text
                result_rows.append(new_row)
            return result_rows

        elif isinstance(transformed, str):
            # Single transformation result
            new_row = row_dict.copy()
            new_row[context_col] = transformed
            return [new_row]

        else:
            # Empty result or unexpected type, keep original
            return [row_dict]

    except Exception as e:
        logger.warning(f"Failed to transform context at row {idx}: {e}")
        return [row_dict]


def process_in_parallel(
    df: pd.DataFrame,
    transformer: Any,
    context_col: str,
    num_processes: int | None = None,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Process dataset in parallel using multiprocessing.

    Args:
        df: Input DataFrame
        transformer: Transformation object
        context_col: Name of context column to transform
        num_processes: Number of processes to use (None for auto-detect)
        chunk_size: Number of rows per chunk

    Returns:
        Complete augmented DataFrame
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    logger.info(
        f"Processing {len(df)} rows using {num_processes} processes "
        f"with chunk size {chunk_size}"
    )

    # Prepare data for parallel processing
    # Convert DataFrame rows to dictionaries for pickling
    row_data = [
        (idx, row.to_dict(), transformer, context_col) for idx, row in df.iterrows()
    ]

    # Process in parallel with progress bar
    all_augmented_rows = []

    with mp.Pool(processes=num_processes) as pool:
        # Use imap for better memory efficiency and progress tracking
        with tqdm(total=len(row_data), desc="Processing rows", unit="row") as pbar:
            for result_rows in pool.imap(
                _transform_row_worker, row_data, chunksize=chunk_size
            ):
                all_augmented_rows.extend(result_rows)
                pbar.update(1)

    # Create DataFrame from all augmented rows
    augmented_df = pd.DataFrame(all_augmented_rows)

    # Reset index and create new sequential ID column if original dataset had one
    augmented_df.reset_index(drop=True, inplace=True)
    if "id" in df.columns:
        augmented_df["id"] = range(1, len(augmented_df) + 1)

    logger.info(
        f"Parallel processing completed: {len(augmented_df)} total rows generated "
        f"from {len(df)} original rows"
    )
    return augmented_df


def transform_context_column(
    df: pd.DataFrame,
    transformer: Any,
    context_col: str = "context",
) -> pd.DataFrame:
    """
    Transform the context column of a dataset using the provided transformer.
    If transformer returns multiple results, creates multiple rows for each original
    row.

    Args:
        df: Input DataFrame containing the dataset
        transformer: Transformation object with a transform method
        context_col: Name of the context column to transform

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

    logger.info(f"Transforming {len(df)} rows in '{context_col}' column")

    # Store all augmented rows
    augmented_rows = []
    total_transformations = 0

    # Create progress bar for transformation process
    with tqdm(total=len(df), desc="Transforming contexts", unit="row") as pbar:
        for idx, row in df.iterrows():
            context = row[context_col]

            if pd.isna(context):
                logger.warning(f"Skipping NaN context at row {idx}")
                # Keep original row for NaN contexts
                new_row = row.copy()
                augmented_rows.append(new_row)
                continue

            try:
                transformed = transformer.transform(str(context))

                # Handle case where transformer returns a list of transformations
                if isinstance(transformed, list) and len(transformed) > 0:
                    # Create multiple rows for each transformation result
                    for _, transformed_text in enumerate(transformed):
                        new_row = row.copy()
                        new_row[context_col] = transformed_text
                        augmented_rows.append(new_row)
                        total_transformations += 1

                elif isinstance(transformed, str):
                    # Single transformation result
                    new_row = row.copy()
                    new_row[context_col] = transformed
                    augmented_rows.append(new_row)
                    total_transformations += 1

                else:
                    # Empty result or unexpected type, keep original
                    logger.warning(
                        f"Unexpected transformation result type at row {idx}: "
                        f"{type(transformed)}"
                    )
                    new_row = row.copy()
                    augmented_rows.append(new_row)

                # Update progress bar with current transformation count
                pbar.set_postfix({"transformations": total_transformations})

            except Exception as e:
                logger.warning(f"Failed to transform context at row {idx}: {e}")
                # Keep original row on failure
                new_row = row.copy()
                augmented_rows.append(new_row)

            # Update progress bar
            pbar.update(1)

    # Create DataFrame from all augmented rows
    augmented_df = pd.DataFrame(augmented_rows)

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

    # Extract parallel processing configuration
    parallel_enabled = config.parallel.get("enabled", False)
    num_processes = config.parallel.get("num_processes", None)
    chunk_size = config.parallel.get("chunk_size", 100)

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

    # Log parallel processing configuration
    if parallel_enabled:
        processes = num_processes or mp.cpu_count()
        logger.info(
            f"Parallel processing enabled: {processes} processes, "
            f"chunk_size={chunk_size}"
        )
    else:
        logger.info("Parallel processing disabled - using sequential processing")

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

    # Transform the dataset - use parallel processing if enabled
    if parallel_enabled:
        augmented_df = process_in_parallel(
            df=df,
            transformer=transformer,
            context_col=context_col,
            num_processes=num_processes,
            chunk_size=chunk_size,
        )
    else:
        # Process sequentially (original behavior)
        augmented_df = transform_context_column(df, transformer, context_col)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save augmented dataset
    augmented_df.to_csv(output_file, index=False)
    logger.info(
        f"Saved augmented dataset with {len(augmented_df)} rows to {output_file}"
    )

    logger.info("Dataset augmentation completed successfully")


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some other platforms
    mp.set_start_method("spawn", force=True)
    main()
