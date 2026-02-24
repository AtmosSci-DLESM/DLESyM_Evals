"""
evaluate_wbx.py

WeatherBench-X Evaluation Script for DLESyM Model Predictions

This script evaluates model predictions against target observations using the WeatherBench-X
framework. It supports evaluation of:
  - Model predictions (normal evaluation)
  - Climatology baseline forecasts (deterministic)
  - Persistence baseline forecasts (deterministic)
  - Probabilistic climatology baseline forecasts (ensemble from historical years)

The script computes metrics (e.g., CRPS, RMSE, ACC) aggregated over spatial regions and
saves results as NetCDF files for downstream analysis and plotting.

Data Loaders:
-------------
This script uses WeatherBench-X built-in data loaders with custom preprocessing for DLESyM format:
  - TargetsFromXarray: For target/analysis datasets (valid_time dimension)
  - ClimatologyFromXarray: For deterministic climatology (dayofyear lookup)
  - PersistenceFromXarray: For persistence baseline (value at init_time for all lead_times)
  - ProbabilisticClimatologyFromXarray: For probabilistic climatology (each year = 1 ensemble member)
  - DLESyMPredictionsFromXarray: Custom loader for DLESyM prediction format (init_time, lead_time)

See: https://weatherbench-x.readthedocs.io/en/latest/api/data_loaders.html

Baseline Types and Metrics:
---------------------------
  | Baseline Type              | Data Loader                        | Metrics       |
  |----------------------------|------------------------------------|---------------|
  | climatology                | ClimatologyFromXarray              | deterministic |
  | persistence                | PersistenceFromXarray              | deterministic |
  | probabilistic_climatology  | ProbabilisticClimatologyFromXarray | probabilistic |

- Deterministic baselines (climatology, persistence) produce a single value per (init_time, lead_time)
  and are evaluated with deterministic metrics (RMSE, Bias, MAE, ACC).
- Probabilistic climatology produces N ensemble members (one per year in [start_year, end_year])
  and can be evaluated with probabilistic metrics (CRPSEnsemble, CRPSSpread, etc.).

Usage Examples:
---------------

1. Normal Model Evaluation:
   ```bash
   python evaluate_wbx.py --config config.yaml --n-jobs 8
   ```

2. Climatology Baseline (deterministic):
   ```bash
   python evaluate_wbx.py --config config.yaml --baseline-type climatology --n-jobs 8
   ```
   Uses climatology dataset specified in config. Evaluates with deterministic metrics only.

3. Persistence Baseline (deterministic):
   ```bash
   python evaluate_wbx.py --config config.yaml --baseline-type persistence --n-jobs 8
   ```
   Uses target dataset as source. Evaluates with deterministic metrics only.

4. Probabilistic Climatology Baseline (ensemble):
   ```bash
   python evaluate_wbx.py --config config.yaml --baseline-type probabilistic_climatology --n-jobs 8
   ```
   Uses target dataset, samples historical years as ensemble members.
   Evaluates with probabilistic metrics (requires config: start_year, end_year).
   
   Config example:
   ```yaml
   baseline_eval:
     start_year: 2000  # First year for ensemble
     end_year: 2020    # Last year (21 ensemble members)
   ```

Output Files:
-------------
The script generates two output files:
  - <output_path>_timeseries.nc: Full timeseries with metrics for each init_time
  - <output_path>.nc: Time-averaged metrics

The output files contain metrics with naming convention:
  - <metric>.<variable> (e.g., "rmse.sst", "crps_ens.t2m")
  - Dimensions: init_time, lead_time, region

Notes:
------
- Command line --baseline-type takes precedence over config file baseline_eval settings
- For climatology baseline, climatology_path must be specified in config
- For persistence/probabilistic_climatology, the target dataset is used as source
- The script processes data in chunks to manage memory (controlled by init_chunk_size)
- Parallelization uses joblib with threading backend (--n-jobs controls worker count)
- Use --max-concurrent-chunks to cap how many chunks run at once (avoids OOM with large n-jobs)
- align_lsm_with_infill (default false): when true, masking uses same LSM and land_threshold as
  training/inference infilling; set masking.lsm_path and masking.land_threshold in config.
"""

import argparse
import logging
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import weatherbenchX as wbx
import weatherbenchX.metrics.deterministic
import weatherbenchX.metrics.probabilistic
import weatherbenchX.weighting
import weatherbenchX.aggregation
from weatherbenchX import binning

# Import custom dataloader factory and climatology processing
from dataloaders.dlesym_dataloader import get_data_loader, process_climatology_dataset, load_land_sea_mask

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress xarray/dataloader warnings to keep the progress bar clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def strip_invalid_fillvalues(ds: xr.Dataset) -> xr.Dataset:
    """Remove or neutralize invalid _FillValue attributes before writing."""
    for name, da in ds.variables.items():
        if "_FillValue" in da.attrs:
            da.attrs.pop("_FillValue", None)
        if np.issubdtype(da.dtype, np.datetime64):
            da.encoding["_FillValue"] = None
    if "_FillValue" in ds.attrs:
        ds.attrs.pop("_FillValue", None)
    return ds

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def chunk_init_times(init_times: np.ndarray, chunk_size: int) -> list:
    """Split init_times into chunks."""
    if chunk_size is None or chunk_size <= 0:
        return [init_times]
    return [init_times[i:i+chunk_size] for i in range(0, len(init_times), chunk_size)]

def extract_init_times(ds: xr.Dataset, start: str, stop: str) -> np.ndarray:
    """Return init times within [start, stop] from dataset coord."""
    init_dim = "init_time" if "init_time" in ds.coords else "time"
    inits = ds[init_dim]
    return inits.sel({init_dim: slice(start, stop)}).values

def get_metric_suite(suite_names: list, config: dict, variables: list = None) -> dict:
    """Build metric suite from config."""
    metrics = {}
    climatology = None
    clim_path = config.get('output', {}).get('climatology_path')
    
    if clim_path:
        try:
            climatology = process_climatology_dataset(clim_path, variables=variables)
        except Exception as e:
            logger.error(f"Failed to load climatology: {e}")
            climatology = None
    
    for name in suite_names:
        if name == "deterministic":
            metrics['rmse'] = wbx.metrics.deterministic.RMSE()
            metrics['bias'] = wbx.metrics.deterministic.Bias()
            metrics['mae'] = wbx.metrics.deterministic.MAE()
            if climatology is not None:
                metrics['acc'] = wbx.metrics.deterministic.ACC(climatology=climatology)
        elif name == "probabilistic":
            metrics['crps_ens'] = wbx.metrics.probabilistic.CRPSEnsemble()
            metrics['spread_skill'] = wbx.metrics.probabilistic.UnbiasedSpreadSkillRatio()
            metrics['crps_skill'] = wbx.metrics.probabilistic.CRPSSkill()
            metrics['ens_var'] = wbx.metrics.probabilistic.EnsembleVariance()
            metrics['rmse_ens'] = wbx.metrics.probabilistic.UnbiasedEnsembleMeanRMSE()
            metrics['rmse'] = wbx.metrics.deterministic.RMSE()
            if climatology is not None:
                metrics['acc'] = wbx.metrics.deterministic.ACC(climatology=climatology)
    
    return metrics

class StaticWeighting:
    def __init__(self, weights: xr.DataArray):
        self._weights = weights
    def weights(self, ds: xr.Dataset) -> xr.DataArray:
        return self._weights

def build_region_binner(region_cfg: dict) -> binning.Regions:
    if not region_cfg:
        return None
    regions = {}
    for name, bounds in region_cfg.items():
        regions[name] = ((bounds["lat_min"], bounds["lat_max"]), (bounds["lon_min"], bounds["lon_max"]))
    return binning.Regions(regions)


def get_output_paths(eval_cfg: dict, baseline_type: str = None, baseline_eval_cfg: dict = None) -> tuple:
    """
    Determine output paths for evaluation results.
    
    Returns:
        tuple: (mean_path, timeseries_path) - paths for time-averaged and full timeseries results
    """
    main_output_path = eval_cfg["output"]["path"]
    
    if baseline_type:
        # Get template path from config (may contain <type> and <forecast_type> placeholders)
        output_path = baseline_eval_cfg.get("output_path") if baseline_eval_cfg else None
        
        # Extract forecast_type from main output path (e.g., "ocean" from "evals_2019-2023_ocean_003.nc")
        main_output_name = os.path.basename(main_output_path).replace(".nc", "")
        forecast_type = "baseline"
        for candidate in ["ocean", "atmosphere", "atmos", "coupled"]:
            if candidate in main_output_name.lower():
                forecast_type = candidate
                break
        
        if output_path:
            # Replace template placeholders
            output_path = output_path.replace("<type>", baseline_type)
            output_path = output_path.replace("<forecast_type>", forecast_type)
        else:
            # Generate default path based on main output path
            base_dir = os.path.dirname(main_output_path)
            base_name = main_output_name.split("_timeseries")[0]
            output_path = os.path.join(base_dir, f"{base_name}_{baseline_type}_timeseries.nc")
        
        mean_path = output_path.replace("_timeseries.nc", ".nc")
        timeseries_path = output_path
    else:
        mean_path = main_output_path
        timeseries_path = mean_path.replace(".nc", "_timeseries.nc")
    
    return mean_path, timeseries_path


# ------------------------------------------------------------------------------
# Parallel Worker Function
# ------------------------------------------------------------------------------

def process_chunk(init_times, lead_times, pred_loader, targ_loader, eval_cfg, metrics_dict):
    """
    Worker function executed in parallel by joblib.
    
    Args:
        init_times: Array of initialization times
        lead_times: Array of lead times
        pred_loader: Prediction data loader
        targ_loader: Target data loader
        eval_cfg: Evaluation configuration dict
        metrics_dict: Dictionary of metrics to compute
    """
    # Load prediction and target data
    ds_pred = pred_loader.load_chunk(init_times, lead_times)
    ds_targ = targ_loader.load_chunk(init_times, lead_times)

    # Align and load eagerly - critical for performance!
    # Note: Land-sea masking is applied in the target dataloader preprocessing
    # ds_pred, ds_targ = xr.align(ds_pred, ds_targ, join="inner")
    ds_pred = ds_pred.load()
    # load xarrays in ds_targ
    if isinstance(ds_targ, dict):
        for k, v in ds_targ.items():
            ds_targ[k] = v.load()

    # Build aggregator for this chunk
    region_binner = build_region_binner(eval_cfg.get("regions"))
    bin_by = [region_binner] if region_binner is not None else None
    
    w_cfg = eval_cfg["aggregation"]["weighting"]
    weights = None
    if w_cfg["class"] == "GridAreaWeighting":
        lat_name = "latitude" if "latitude" in ds_pred.coords else w_cfg.get("latitude_name", "latitude")
        base_weighting = wbx.weighting.GridAreaWeighting(latitude_name=lat_name)
        weights = base_weighting.weights(ds_pred)

    aggregator = wbx.aggregation.Aggregator(
        reduce_dims=eval_cfg["aggregation"].get("reduce_dims", ["latitude", "longitude"]),
        weigh_by=[StaticWeighting(weights)] if weights is not None else None,
        bin_by=bin_by,
        masked=True,
        skipna=False,
    )

    # Compute metrics
    chunk_ds = wbx.aggregation.compute_metric_values_for_single_chunk(
        metrics_dict,
        aggregator,
        ds_pred,
        ds_targ,
    )
    return chunk_ds

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--baseline-type", choices=["climatology", "persistence", "probabilistic_climatology"], default=None)
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel jobs")
    parser.add_argument(
        "--max-concurrent-chunks",
        type=int,
        default=None,
        help="Max chunks to process in parallel (limits memory). Default: from config or 8. Set to avoid OOM when using large n-jobs.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    eval_cfg = config["evaluation"]

    # Baseline Logic
    baseline_type = args.baseline_type
    baseline_eval_cfg = eval_cfg.get("baseline_eval", {})
    if baseline_eval_cfg.get("enabled", False) and baseline_type is None:
        baseline_type = baseline_eval_cfg.get("type")

    # Get masking config to pass to target dataloader.
    # When align_lsm_with_infill is True, build masking from infill-aligned LSM path and
    # land_threshold so that evaluation excludes the same grid points as infilling (backward
    # compatible: default False).
    align_lsm = eval_cfg.get("align_lsm_with_infill", False)
    masking_block = eval_cfg.get("masking", {})
    if align_lsm:
        lsm_path = masking_block.get("lsm_path") or masking_block.get("path")
        land_threshold = masking_block.get("land_threshold")
        if land_threshold is None and "threshold" in masking_block:
            land_threshold = masking_block["threshold"]
        if lsm_path and land_threshold is not None:
            masking_cfg = {
                "enabled": True,
                "path": lsm_path,
                "variable": masking_block.get("variable", "lsm"),
                "threshold": land_threshold,
                "type": "land",
            }
            logger.info(
                "align_lsm_with_infill=True: using infill-aligned masking (path=%s, land_threshold=%s)",
                lsm_path, land_threshold,
            )
            keep_mask = load_land_sea_mask(masking_cfg)
            if keep_mask is not None:
                land_frac = 1.0 - float(keep_mask.mean())
                logger.info("Land fraction (masked out for eval): %.4f", land_frac)
        else:
            logger.warning(
                "align_lsm_with_infill=True but masking.lsm_path/path or masking.land_threshold missing; "
                "falling back to existing masking config."
            )
            masking_cfg = masking_block
    else:
        masking_cfg = masking_block

    # 1. Initialize Loaders (using WBX built-in loaders where possible)
    if baseline_type == "climatology":
        # Deterministic climatology: dayofyear lookup, single value per (init_time, lead_time)
        clim_path = baseline_eval_cfg.get("climatology_path") or eval_cfg.get("output", {}).get("climatology_path")
        pred_config = {
            "class": "ClimatologyFromXarray",  # WBX built-in loader
            "path": clim_path,
            "variables": eval_cfg["data"].get("prediction", {}).get("variables") or eval_cfg["data"].get("target", {}).get("variables"),
            "compute": True,
            "add_nan_mask": masking_cfg.get("add_nan_mask", False),
        }
        pred_loader = get_data_loader(pred_config)
        # Add masking config to target loader
        target_config = eval_cfg["data"]["target"].copy()
        target_config["masking"] = masking_cfg
        targ_loader = get_data_loader(target_config)
    elif baseline_type == "persistence":
        # Deterministic persistence: value at init_time for all lead_times
        pred_config = {
            "class": "PersistenceFromXarray",  # WBX built-in loader
            "path": eval_cfg["data"]["target"]["path"],
            "variables": eval_cfg["data"].get("prediction", {}).get("variables") or eval_cfg["data"].get("target", {}).get("variables"),
            "compute": True,
            "add_nan_mask": masking_cfg.get("add_nan_mask", False),
        }
        pred_loader = get_data_loader(pred_config)
        # Add masking config to target loader
        target_config = eval_cfg["data"]["target"].copy()
        target_config["masking"] = masking_cfg
        targ_loader = get_data_loader(target_config)
    elif baseline_type == "probabilistic_climatology":
        # Probabilistic climatology: each year in [start_year, end_year] is an ensemble member
        # Uses target dataset to sample historical years for valid_time's dayofyear
        start_year = baseline_eval_cfg.get("start_year", 2000)
        end_year = baseline_eval_cfg.get("end_year", 2020)
        pred_config = {
            "class": "ProbabilisticClimatologyFromXarray",  # WBX built-in loader
            "path": eval_cfg["data"]["target"]["path"],  # Uses target dataset!
            "variables": eval_cfg["data"].get("prediction", {}).get("variables") or eval_cfg["data"].get("target", {}).get("variables"),
            "start_year": start_year,
            "end_year": end_year,
            "compute": True,
            "add_nan_mask": masking_cfg.get("add_nan_mask", False),
        }
        pred_loader = get_data_loader(pred_config)
        # Add masking config to target loader
        target_config = eval_cfg["data"]["target"].copy()
        target_config["masking"] = masking_cfg
        targ_loader = get_data_loader(target_config)
        logger.info(f"Using probabilistic climatology with years {start_year}-{end_year} ({end_year - start_year + 1} ensemble members)")
    else:
        pred_loader = get_data_loader(eval_cfg["data"]["prediction"])
        # Add masking config to target loader
        target_config = eval_cfg["data"]["target"].copy()
        target_config["masking"] = masking_cfg
        targ_loader = get_data_loader(target_config)

    # WBX loaders use lazy loading - initialize the underlying dataset
    if hasattr(targ_loader, 'maybe_prepare_dataset'):
        targ_loader.maybe_prepare_dataset()
    if hasattr(pred_loader, 'maybe_prepare_dataset'):
        pred_loader.maybe_prepare_dataset()

    variables = eval_cfg["data"].get("prediction", {}).get("variables") or eval_cfg["data"].get("target", {}).get("variables")
    
    # Force deterministic metrics for deterministic baselines (climatology, persistence)
    # Probabilistic metrics like CRPSSpread require multiple ensemble members
    if baseline_type in ["climatology", "persistence"]:
        logger.info(f"Baseline type '{baseline_type}' is deterministic - using deterministic metrics only")
        metrics_suites = ["deterministic"]
    else:
        metrics_suites = eval_cfg["metrics"]
    
    metrics_dict = get_metric_suite(metrics_suites, eval_cfg, variables=variables)

    # 2. Get Init Times
    if baseline_type in ["climatology", "persistence", "probabilistic_climatology"]:
        valid_dim = "valid_time" if "valid_time" in targ_loader._ds.dims else "time"
        inits = targ_loader._ds[valid_dim]
        init_times_all = inits.sel({valid_dim: slice(eval_cfg["time_start"], eval_cfg["time_stop"])}).values
    else:
        init_times_all = extract_init_times(pred_loader._ds, eval_cfg["time_start"], eval_cfg["time_stop"])

    # --- ENFORCE CHUNKING ---
    chunk_size = eval_cfg.get("init_chunk_size")
    if not chunk_size:
        chunk_size = 2  # Default to 10 to ensure parallelism works
        logger.info(f"init_chunk_size not set. Defaulting to {chunk_size} for parallel processing.")
    
    init_chunks = chunk_init_times(init_times_all, chunk_size)

    # 3. Get Lead Times
    if baseline_type in ["climatology", "persistence", "probabilistic_climatology"]:
        # Check if lead_times are specified in baseline_eval config
        lead_times_cfg = baseline_eval_cfg.get("lead_times")
        if lead_times_cfg:
            start = lead_times_cfg.get("start", 0)
            stop = lead_times_cfg.get("stop", 97)
            step = lead_times_cfg.get("step", 6)
            unit = lead_times_cfg.get("unit", "hours")
            
            if unit == "days":
                lead_times = np.arange(start, stop, step, dtype="timedelta64[D]")
            else:  # default to hours
                lead_times = np.arange(start, stop, step, dtype="timedelta64[h]")
            logger.info(f"Using configured lead_times: {start}-{stop} {unit} in steps of {step}")
        elif "lead_time" in targ_loader._ds.coords:
            lead_times = targ_loader._ds["lead_time"].values
            logger.info(f"Using lead_times from target dataset: {len(lead_times)} values")
        else:
            lead_times = np.arange(0, 97, 6, dtype="timedelta64[h]")
            logger.info(f"Using default lead_times: 0-96 hours in 6-hour steps")
    else:
        lead_times = pred_loader._ds["lead_time"].values if "lead_time" in pred_loader._ds.coords else np.array([0], dtype="timedelta64[ns]")

    # 4. RUN EVALUATION
    # Limit concurrent chunks to avoid OOM: each worker loads pred+target for one chunk.
    max_concurrent = args.max_concurrent_chunks or eval_cfg.get("max_concurrent_chunks") or 8
    n_jobs = min(args.n_jobs, max_concurrent)
    logger.info(f"Starting evaluation over {len(init_chunks)} chunks (n_jobs={n_jobs}, max_concurrent_chunks={max_concurrent})...")

    # Fix for InvalidIndexError: Force the index hash tables to build sequentially
    # in the main thread before forking/threading.
    logger.info("Priming indices for parallel execution...")
    try:
        if hasattr(pred_loader, '_ds') and "valid_time" in pred_loader._ds.coords:
            # Doing a dummy selection forces the index engine to build
            _ = pred_loader._ds.sel(valid_time=pred_loader._ds.valid_time.values[0], method=None)
            
        if hasattr(targ_loader, '_ds') and "valid_time" in targ_loader._ds.coords:
            # Do the same for the target loader
            _ = targ_loader._ds.sel(valid_time=targ_loader._ds.valid_time.values[0], method=None)
    except Exception as e:
        logger.warning(f"Could not prime indices (this might be fine if dataset is small): {e}")
    
    # Process in batches of max_concurrent chunks to cap memory (each worker holds a full chunk in memory).
    # Use threading backend to avoid pickle serialization overhead of large xarray datasets.
    results = []
    batch_starts = list(range(0, len(init_chunks), max_concurrent))
    for start in tqdm(batch_starts, desc="Evaluating (batches)", unit="batch"):
        batch = init_chunks[start : start + max_concurrent]
        batch_jobs = min(n_jobs, len(batch))
        results_batch = Parallel(n_jobs=batch_jobs, backend="threading")(
            delayed(process_chunk)(
                chunk, lead_times, pred_loader, targ_loader, eval_cfg, metrics_dict
            ) for chunk in batch
        )
        results.extend(results_batch)

    # Filter out failed chunks (None)
    valid_results = [res for res in results if res is not None]

    # 5. Save Results
    if valid_results:
        logger.info("Concatenating and saving...")
        full_ds = xr.concat(valid_results, dim="init_time")
        
        # Determine output paths
        mean_path, timeseries_path = get_output_paths(eval_cfg, baseline_type, baseline_eval_cfg)

        # Save time-averaged results
        full_ds_mean = strip_invalid_fillvalues(full_ds.mean(dim="init_time"))
        full_ds_mean.to_netcdf(mean_path)
        logger.info(f"Saved time-averaged results to {mean_path}")
        
        # Save full timeseries
        full_ds = strip_invalid_fillvalues(full_ds)
        full_ds.to_netcdf(timeseries_path)
        logger.info(f"Saved timeseries results to {timeseries_path}")
    else:
        logger.error("No valid results produced.")

if __name__ == "__main__":
    main()