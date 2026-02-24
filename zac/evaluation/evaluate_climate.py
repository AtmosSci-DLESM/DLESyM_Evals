"""
evaluate_climate.py

Climate Evaluation Script for Climate Rollouts

This script evaluates climate model forecasts by running a series of analyses
defined in configuration files. Each analysis is implemented as a function in
climate_utils/ and can be enabled/disabled via config.

The script loads both a climate inference file (model predictions) and a target
file (observations/reference data) for comparison. It iterates through each enabled
analysis in the config, calling the corresponding function with both datasets
and kwargs from the config.

Usage:
    # Using defaults (ocean_climate_config.yaml and paths from config):
    python evaluate_climate.py
    
    # Specifying config file:
    python evaluate_climate.py --config atmos_climate_config.yaml
    
    # Overriding paths via command line:
    python evaluate_climate.py --inference-file path/to/inference.zarr --target-file path/to/target.zarr
    
    # Full example:
    python evaluate_climate.py --config ocean_climate_config.yaml --inference-file path/to/inference.zarr --target-file path/to/target.zarr

Configuration:
    Config files (atmos_climate_config.yaml or ocean_climate_config.yaml) contain:
        - inference_path: Optional path to inference dataset (if not provided via --inference-file)
        - target_path: Path to target/reference dataset (observations)
        - masking: Optional masking configuration for target dataset
        - analyses: List of analysis dictionaries, each with:
            - name: Descriptive name for the analysis
            - function: Function name in climate_utils/ to call
            - enabled: Boolean to enable/disable the analysis
            - kwargs: Dictionary of keyword arguments to pass to the function
    
    Defaults:
        - If --config is not provided, defaults to ocean_climate_config.yaml
        - If --inference-file is not provided, uses inference_path from config
        - If --target-file is not provided, uses target_path from config

Example Config:
    target_path: "/path/to/target.zarr"
    masking:
      enabled: true
      type: "land"
      path: "/path/to/lsm.nc"
      variable: "lsm"
      threshold: 0.5
    analyses:
      - name: "global_sst_timeseries"
        function: "plot_global_sst_timeseries"
        enabled: true
        kwargs:
          variable: "sst"
          output_path: "outputs/global_sst_timeseries.png"

Notes:
    - All analysis functions should accept (inference_file: xr.Dataset, target_file: xr.Dataset, **kwargs)
    - Functions are dynamically imported from climate_utils.{function_name}
    - Errors in one analysis do not stop execution of other analyses
    - Both datasets are preprocessed to standardize coordinates (lat/lon -> latitude/longitude)
    - Target dataset can optionally have land-sea masking applied
"""

import argparse
import importlib
import logging
import os
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import xarray as xr
import yaml
import zarr

# Import preprocessing utilities
from dataloaders.dlesym_dataloader import (
    ensure_latitude_ascending,
    preprocess_dlesym_predictions,
    preprocess_dlesym_targets,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress xarray warnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def consolidate_zarr_metadata(path: str) -> bool:
    """Attempt to consolidate zarr metadata for faster opening."""
    try:
        g = zarr.open_group(path, mode="r")
        zarr.consolidate_metadata(path)
        return True
    except Exception as e:
        logger.debug(f"Could not consolidate zarr metadata: {e}")
        return False


def load_climate_inference(path: str, variables: Optional[List[str]] = None) -> xr.Dataset:
    """
    Load and preprocess climate inference file.
    
    Handles coordinate renaming (lat/lon -> latitude/longitude), unpacks
    channel dimensions if present, and ensures latitude is ascending.
    Similar to DLESyMPredictionsFromXarray preprocessing.
    
    Args:
        path: Path to inference zarr file
        variables: Optional list of variables to load
    
    Returns:
        Preprocessed xarray Dataset
    
    Raises:
        FileNotFoundError: If the inference file does not exist
        ValueError: If requested variables are not found in the dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inference file not found: {path}")
    
    logger.info(f"Loading climate inference file from: {path}")
    
    # Attempt to consolidate zarr metadata for faster opening
    if path.endswith('.zarr'):
        consolidate_zarr_metadata(path)
        ds = xr.open_zarr(path, consolidated=True)
    else:
        ds = xr.open_dataset(path)
    
    # Apply preprocessing (renames lat/lon, ensures latitude ascending)
    ds = preprocess_dlesym_predictions(ds)
    
    # Select variables if specified
    if variables:
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            logger.warning(f"Variables {missing} not found in inference dataset. Available: {list(ds.data_vars)}")
        available_vars = [v for v in variables if v in ds.data_vars]
        if not available_vars:
            raise ValueError(f"None of the requested variables {variables} found in dataset.")
        ds = ds[available_vars]
    
    # Handle ensemble dimension (rename to 'number' if present, or remove if single member)
    if 'ensemble' in ds.dims:
        if ds.sizes['ensemble'] == 1:
            # Single ensemble member - squeeze it out
            ds = ds.isel(ensemble=0).squeeze()
            logger.info("Removed single ensemble dimension")
        else:
            # Multiple members - rename to 'number' (WBX convention)
            ds = ds.rename_dims({"ensemble": "number"})
            if 'ensemble' in ds.coords:
                ds = ds.rename({"ensemble": "number"})
            logger.info(f"Renamed ensemble dimension to 'number' ({ds.sizes['number']} members)")
    
    # Handle init_time dimension - for climate forecasts, we may want to convert to time
    # If we have init_time + lead_time, we can derive a time dimension
    if "init_time" in ds.dims and "lead_time" in ds.dims:
        if ds.sizes["init_time"] == 1:
            # Single init time - convert to time dimension
            init_time = ds["init_time"].values[0]
            lead_times = ds["lead_time"].values
            
            # Create time coordinate from init_time + lead_time
            if np.issubdtype(lead_times.dtype, np.timedelta64):
                time_values = init_time + lead_times
            else:
                # Assume lead_time is in hours if not timedelta
                time_values = init_time + np.array([np.timedelta64(int(lt), 'h') for lt in lead_times])
            
            # Create new dataset with time dimension
            ds = ds.isel(init_time=0).squeeze()
            ds = ds.assign_coords(time=("lead_time", time_values))
            ds = ds.swap_dims({"lead_time": "time"})
            logger.info("Converted init_time + lead_time to time dimension")
    
    logger.info(f"Loaded inference file. Variables: {list(ds.data_vars)}, Dimensions: {dict(ds.dims)}")
    
    return ds


def load_climate_target(
    path: str,
    variables: Optional[List[str]] = None,
    masking_cfg: Optional[Dict[str, Any]] = None
) -> xr.Dataset:
    """
    Load and preprocess climate target/reference file (observations).
    
    Handles coordinate renaming (lat/lon -> latitude/longitude, time -> valid_time),
    unpacks channel dimensions if present, and optionally applies land-sea masking.
    Similar to preprocess_dlesym_targets preprocessing.
    
    Args:
        path: Path to target zarr file
        variables: Optional list of variables to load
        masking_cfg: Optional masking configuration dict for land-sea masking
    
    Returns:
        Preprocessed xarray Dataset
    
    Raises:
        FileNotFoundError: If the target file does not exist
        ValueError: If requested variables are not found in the dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target file not found: {path}")
    
    logger.info(f"Loading climate target file from: {path}")
    
    # Attempt to consolidate zarr metadata for faster opening
    if path.endswith('.zarr'):
        consolidate_zarr_metadata(path)
        ds = xr.open_zarr(path, consolidated=True)
    else:
        ds = xr.open_dataset(path)
    
    # Apply preprocessing (renames lat/lon, time->valid_time, unpacks channels, applies mask)
    ds = preprocess_dlesym_targets(ds, masking_cfg=masking_cfg)
    
    # Select variables if specified
    if variables:
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            logger.warning(f"Variables {missing} not found in target dataset. Available: {list(ds.data_vars)}")
        available_vars = [v for v in variables if v in ds.data_vars]
        if not available_vars:
            raise ValueError(f"None of the requested variables {variables} found in dataset.")
        ds = ds[available_vars]
    
    # Handle time dimension - for target datasets, we typically have valid_time
    # Convert valid_time to time if it's the only time dimension
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
        logger.info("Renamed valid_time to time dimension")
    
    logger.info(f"Loaded target file. Variables: {list(ds.data_vars)}, Dimensions: {dict(ds.dims)}")
    
    return ds


def run_analysis(
    analysis_cfg: Dict[str, Any],
    inference_file: xr.Dataset,
    target_file: xr.Dataset
) -> bool:
    """
    Run a single analysis function.
    
    Dynamically imports the function from climate_utils.{function_name} and
    calls it with both the inference and target datasets and kwargs from config.
    
    Args:
        analysis_cfg: Analysis configuration dict with keys:
            - name: str - Descriptive name
            - function: str - Function name in climate_utils/
            - enabled: bool - Whether to run this analysis
            - kwargs: dict - Keyword arguments to pass to function
        inference_file: Preprocessed inference dataset (model predictions)
        target_file: Preprocessed target dataset (observations/reference)
    
    Returns:
        True if successful, False otherwise
    """
    name = analysis_cfg.get("name", "unknown")
    function_name = analysis_cfg.get("function")
    enabled = analysis_cfg.get("enabled", True)
    kwargs = analysis_cfg.get("kwargs", {})
    
    if not enabled:
        logger.info(f"Skipping disabled analysis: {name}")
        return True
    
    if not function_name:
        logger.error(f"Analysis '{name}' has no function specified")
        return False
    
    try:
        logger.info(f"Running analysis: {name} (function: {function_name})")
        
        # Dynamically import the function from climate_utils
        # Try to map function name to module name (e.g., "plot_global_sst_timeseries" -> "global_sst_timeseries")
        # First try the function name as-is, then try removing common prefixes
        module_names_to_try = [function_name]
        
        # Remove common prefixes to get module name
        prefixes = ["plot_", "analyze_", "compute_", "generate_"]
        for prefix in prefixes:
            if function_name.startswith(prefix):
                module_names_to_try.append(function_name[len(prefix):])
                break
        
        module = None
        for module_name in module_names_to_try:
            try:
                module = importlib.import_module(f"climate_utils.{module_name}")
                logger.debug(f"Successfully imported module: climate_utils.{module_name}")
                break
            except ImportError:
                continue
        
        if module is None:
            logger.error(f"Failed to import module for function '{function_name}'. Tried: {module_names_to_try}")
            return False
        
        # Get the function from the module
        try:
            analysis_function = getattr(module, function_name)
        except AttributeError:
            # Try to find a function with a similar name (e.g., if module has "plot_global_sst" but we want "plot_global_sst_timeseries")
            # List all functions in the module and suggest
            available = [name for name in dir(module) if not name.startswith("_") and callable(getattr(module, name))]
            logger.error(
                f"Function '{function_name}' not found in module 'climate_utils.{module_names_to_try[0]}'. "
                f"Available functions: {available}"
            )
            return False
        
        # Call the analysis function with both datasets
        result = analysis_function(inference_file, target_file, **kwargs)
        
        logger.info(f"Successfully completed analysis: {name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed analysis '{name}': {e}", exc_info=True)
        return False


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate climate forecasts using analyses defined in config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ocean_climate_config.yaml",
        help="Path to config file (atmos_climate_config.yaml or ocean_climate_config.yaml). Defaults to ocean_climate_config.yaml"
    )
    parser.add_argument(
        "--inference-file",
        type=str,
        default=None,
        help="Path to climate inference zarr file (model predictions). If not provided, uses inference_path from config."
    )
    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help="Path to target zarr file (observations/reference). If not provided, uses target_path from config."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for analysis results (overrides config paths)"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of variables to load from datasets"
    )
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    logger.info(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate config structure
    if "analyses" not in config:
        logger.error("Config file must contain 'analyses' key with list of analysis dictionaries")
        return
    
    analyses = config["analyses"]
    if not isinstance(analyses, list):
        logger.error("'analyses' must be a list of analysis dictionaries")
        return
    
    logger.info(f"Found {len(analyses)} analysis configurations")
    
    # Get inference file path (from command line or config)
    inference_path = args.inference_file or config.get("inference_path")
    if not inference_path:
        logger.error("Inference file path must be provided via --inference-file or inference_path in config")
        return
    
    # Get target file path (from command line or config)
    target_path = args.target_file or config.get("target_path")
    if not target_path:
        logger.error("Target file path must be provided via --target-file or target_path in config")
        return
    
    # Get masking config for target dataset
    masking_cfg = config.get("masking")
    
    # Load and preprocess inference file
    try:
        inference_file = load_climate_inference(inference_path, variables=args.variables)
    except Exception as e:
        logger.error(f"Failed to load inference file: {e}")
        return
    
    # Load and preprocess target file
    try:
        target_file = load_climate_target(target_path, variables=args.variables, masking_cfg=masking_cfg)
    except Exception as e:
        logger.error(f"Failed to load target file: {e}")
        return
    
    # Override output paths if output-dir is specified
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {args.output_dir}")
        # Update kwargs in each analysis config to use output_dir
        for analysis_cfg in analyses:
            if "kwargs" in analysis_cfg and "output_path" in analysis_cfg["kwargs"]:
                # Make path relative to output_dir
                original_path = analysis_cfg["kwargs"]["output_path"]
                basename = os.path.basename(original_path)
                analysis_cfg["kwargs"]["output_path"] = os.path.join(args.output_dir, basename)
    
    # Run each enabled analysis
    results = []
    for i, analysis_cfg in enumerate(analyses):
        success = run_analysis(analysis_cfg, inference_file, target_file)
        results.append((analysis_cfg.get("name", f"analysis_{i}"), success))
    
    # Summary
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    logger.info("=" * 60)
    logger.info("Analysis Summary:")
    logger.info(f"  Total analyses: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    
    if failed > 0:
        logger.info("\nFailed analyses:")
        for name, success in results:
            if not success:
                logger.info(f"  - {name}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

