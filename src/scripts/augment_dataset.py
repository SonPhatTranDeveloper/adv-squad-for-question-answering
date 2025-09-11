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

BATCH PROCESSING FEATURES:
- Process large datasets in configurable batches to manage memory usage
- Save intermediate results for recovery in case of interruption
- Memory optimization with garbage collection between batches
- Detailed progress tracking for batch processing

To enable batch processing, set batch.enabled=true in your config file.
See dataset_augment_batch_example.yaml for an example configuration.
"""

import gc
import logging
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


def transform_context_column(
    df: pd.DataFrame,
    transformer: Any,
    context_col: str = "context",
    batch_info: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    Transform the context column of a dataset using the provided transformer.
    If transformer returns multiple results, creates multiple rows for each original
    row.

    Args:
        df: Input DataFrame containing the dataset
        transformer: Transformation object with a transform method
        context_col: Name of the context column to transform
        batch_info: Optional tuple of (batch_number, total_batches)
        for progress tracking

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

    if batch_info:
        batch_num, total_batches = batch_info
        logger.info(
            f"Transforming batch {batch_num}/{total_batches}: {len(df)} rows in "
            f"{context_col}' column"
        )
    else:
        logger.info(f"Transforming {len(df)} rows in '{context_col}' column")

    # Store all augmented rows
    augmented_rows = []
    total_transformations = 0

    # Create progress bar for transformation process
    desc = (
        f"Batch {batch_info[0]}/{batch_info[1]}"
        if batch_info
        else "Transforming contexts"
    )
    with tqdm(total=len(df), desc=desc, unit="row") as pbar:
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


def process_in_batches(
    df: pd.DataFrame,
    transformer: Any,
    context_col: str,
    batch_size: int,
    save_intermediate: bool = False,
    intermediate_dir: Path | None = None,
    output_file_base: str | None = None,
    memory_efficient: bool = True,
) -> pd.DataFrame:
    """
    Process dataset in batches to handle large datasets efficiently.

    Args:
        df: Input DataFrame
        transformer: Transformation object
        context_col: Name of context column to transform
        batch_size: Number of rows per batch
        save_intermediate: Whether to save intermediate batch results
        intermediate_dir: Directory to save intermediate results
        output_file_base: Base name for output files (without extension)
        memory_efficient: Whether to perform garbage collection after each batch

    Returns:
        Complete augmented DataFrame
    """
    total_batches = (len(df) + batch_size - 1) // batch_size
    logger.info(
        f"Processing {len(df)} rows in {total_batches} batches of size {batch_size}"
    )

    # Create intermediate directory if needed
    if save_intermediate and intermediate_dir:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Intermediate results will be saved to: {intermediate_dir}")

    all_augmented_rows = []

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()

        logger.info(
            f"Processing batch {batch_idx + 1}/{total_batches} "
            f"(rows {start_idx}-{end_idx - 1})"
        )

        # Transform current batch
        batch_augmented = transform_context_column(
            batch_df,
            transformer,
            context_col,
            batch_info=(batch_idx + 1, total_batches),
        )

        # Save intermediate results if requested
        if save_intermediate and intermediate_dir and output_file_base:
            batch_filename = f"{output_file_base}_batch_{batch_idx + 1:03d}.csv"
            batch_file_path = intermediate_dir / batch_filename
            batch_augmented.to_csv(batch_file_path, index=False)
            logger.info(f"Saved batch {batch_idx + 1} results to: {batch_file_path}")

        all_augmented_rows.append(batch_augmented)

        # Log memory usage info and clean up
        logger.info(
            f"Batch {batch_idx + 1} completed: {len(batch_augmented)} rows generated"
        )

        # Force garbage collection after each batch to manage memory (if enabled)
        if memory_efficient:
            del batch_df, batch_augmented
            gc.collect()

    # Combine all batches
    logger.info("Combining all batch results...")
    final_df = pd.concat(all_augmented_rows, ignore_index=True)

    # Reset ID column if it exists
    if "id" in df.columns:
        final_df["id"] = range(1, len(final_df) + 1)

    logger.info(f"Batch processing completed: {len(final_df)} total rows generated")
    return final_df


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

    # Extract batch processing configuration
    batch_enabled = config.batch.get("enabled", False)
    batch_size = config.batch.get("size", 1000)
    save_intermediate = config.batch.get("save_intermediate", False)
    memory_efficient = config.batch.get("memory_efficient", True)
    intermediate_dir = Path(config.batch.get("intermediate_dir", "outputs/batches"))

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

    # Log batch processing configuration
    if batch_enabled:
        logger.info(
            f"Batch processing enabled: batch_size={batch_size}, "
            f"save_intermediate={save_intermediate}"
        )
        if save_intermediate:
            logger.info(f"Intermediate files will be saved to: {intermediate_dir}")
    else:
        logger.info("Batch processing disabled - processing entire dataset at once")

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

    # Transform the dataset - use batch processing if enabled
    if batch_enabled:
        # Extract base name for intermediate files
        output_base_name = output_file.stem

        # Process in batches
        augmented_df = process_in_batches(
            df=df,
            transformer=transformer,
            context_col=context_col,
            batch_size=batch_size,
            save_intermediate=save_intermediate,
            intermediate_dir=intermediate_dir if save_intermediate else None,
            output_file_base=output_base_name if save_intermediate else None,
            memory_efficient=memory_efficient,
        )
    else:
        # Process entire dataset at once (original behavior)
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
    main()
