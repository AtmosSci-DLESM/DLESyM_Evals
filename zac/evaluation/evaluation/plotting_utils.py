"""
plotting_utils.py

Shared utilities for plotting scripts (skill_curves.py and scorecard_utils.py).
Contains common functions for loading models from config and variable detection/sorting.
"""

import yaml
import os
import xarray as xr

# Define standard field ordering
OCEAN_FIELDS = [
    "sit", "sst", "sic", "ssh", "s0m", "s10m", "s50m", "s100m", "s200m", 
    "t0m", "t5m", "t10m", "t25m", "t37.5m", "t50m", "t62.5m", "t75m", "t87.5m", 
    "t100m", "t125m", "t150m", "t200m", "t300m"
]

ATMOS_FIELDS = [
    "t2m", "t850", "tau300-700", "tcwv", "ws10", "z250", "z500", "z1000", "ttr"
]


def load_models_from_config(config_path):
    """
    Load models from a plotting config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        tuple: (experiments_dict, baseline_label, baseline_dataset)
            - experiments_dict: Dictionary mapping model labels to xarray Datasets
            - baseline_label: Label of the baseline model (or None)
            - baseline_dataset: xarray Dataset for baseline (or None)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    eval_base_dir = config.get('eval_base_dir', '/pscratch/sd/z/zespinos/crps_deterministic_evals')
    forecast_type = config.get('forecast_type', 'ocean')
    models_config = config.get('models', [])
    baseline_label = config.get('baseline', None)
    
    experiments = {}
    failed_models = []
    baseline_ds = None
    
    for model_cfg in models_config:
        label = model_cfg.get('label', 'Unknown')
        xid = model_cfg.get('xid', '')
        
        # Determine path
        if 'path' in model_cfg and model_cfg['path']:
            eval_path = model_cfg['path']
        else:
            # Construct path from base dir, forecast type, and xid
            eval_path = os.path.join(eval_base_dir, f"evals_{forecast_type}_{xid}_timeseries.nc")
        
        # Try to load the dataset
        try:
            if not os.path.exists(eval_path):
                print(f"WARNING: Evaluation file not found for {label} at {eval_path}, skipping...")
                failed_models.append(label)
                continue
            
            ds = xr.open_dataset(eval_path)
            experiments[label] = ds
            print(f"Loaded model: {label} from {eval_path}")
            
            # Check if this is the baseline
            if label == baseline_label:
                baseline_ds = ds
        except Exception as e:
            print(f"ERROR: Failed to load {label} from {eval_path}: {e}")
            failed_models.append(label)
            continue
    
    if not experiments:
        raise ValueError("No models could be loaded from config. Check file paths and XIDs.")
    
    if failed_models:
        print(f"WARNING: Failed to load {len(failed_models)} model(s): {', '.join(failed_models)}")
    
    if baseline_label and baseline_ds is None:
        # Check if baseline is a model that failed to load
        baseline_found = any(model_cfg.get('label') == baseline_label for model_cfg in models_config)
        if baseline_found:
            raise ValueError(f"Baseline model '{baseline_label}' not found in loaded models. Check config.")
        # If not found in models, it might be climatology/persistence - that's OK
    
    # Check for baseline config (climatology/persistence)
    if baseline_ds is None:
        baseline_ds, baseline_label = load_baseline(
            config=config,
            eval_base_dir=eval_base_dir,
            forecast_type=forecast_type
        )
    
    return experiments, baseline_label, baseline_ds


def load_baseline(baseline_path=None, baseline_label=None, config=None, 
                  eval_base_dir=None, forecast_type=None):
    """
    Load baseline dataset (climatology or persistence) from path or config.
    
    Args:
        baseline_path (str, optional): Direct path to baseline evaluation file (command line).
        baseline_label (str, optional): Label for baseline (command line).
        config (dict, optional): Full config dictionary (for config-based loading).
        eval_base_dir (str, optional): Base directory for evaluation files (from config).
        forecast_type (str, optional): Forecast type 'ocean' or 'atmos' (from config).
        
    Returns:
        tuple: (baseline_ds, baseline_label)
            - baseline_ds: xarray Dataset or None
            - baseline_label: str or None
    """
    baseline_ds = None
    final_label = baseline_label
    
    # Priority 1: Load from command line path if provided
    if baseline_path:
        if os.path.exists(baseline_path):
            baseline_ds = xr.open_dataset(baseline_path)
            if final_label is None:
                # Auto-detect label from filename
                basename = os.path.basename(baseline_path)
                if "climatology" in basename.lower():
                    final_label = "Climatology"
                elif "persistence" in basename.lower():
                    final_label = "Persistence"
                else:
                    final_label = "Baseline"
            print(f"Loaded baseline: {final_label} from {baseline_path}")
        else:
            print(f"WARNING: Baseline file not found at {baseline_path}")
        return baseline_ds, final_label
    
    # Priority 2: Load from config if enabled
    if config:
        baseline_cfg = config.get('baseline') or {}  # Handle None explicitly
        if baseline_cfg.get('enabled', False):
            config_path = baseline_cfg.get('path')
            if config_path:
                # Auto-construct path if type is specified and path is relative
                if 'type' in baseline_cfg and not config_path.startswith('/'):
                    if eval_base_dir is None:
                        eval_base_dir = config.get('eval_base_dir', '/pscratch/sd/z/zespinos/crps_deterministic_evals')
                    if forecast_type is None:
                        forecast_type = config.get('forecast_type', 'ocean')
                    baseline_type = baseline_cfg['type']
                    config_path = os.path.join(eval_base_dir, f"evals_{forecast_type}_{baseline_type}_timeseries.nc")
                
                if os.path.exists(config_path):
                    baseline_ds = xr.open_dataset(config_path)
                    final_label = baseline_cfg.get('label', final_label or baseline_cfg.get('type', 'Baseline').title())
                    print(f"Loaded baseline: {final_label} from {config_path}")
                else:
                    print(f"WARNING: Baseline file not found at {config_path}")
    
    return baseline_ds, final_label


def detect_and_sort_variables(ds, metric):
    """
    Detect variable type (ocean vs atmosphere) and sort variables based on standard ordering.
    
    Args:
        ds: xarray Dataset
        metric: Metric prefix (e.g., 'crps_ens')
        
    Returns:
        tuple: (obs_vars, is_ocean)
            - obs_vars: Sorted list of variable names
            - is_ocean: Boolean indicating if variables are ocean type
    """
    raw_vars = [v for v in ds.data_vars if v.startswith(f"{metric}.")]
    
    if not raw_vars:
        raise ValueError(f"No variables found starting with '{metric}'")
    
    prefix_len = len(metric) + 1
    suffixes = [v[prefix_len:] for v in raw_vars]
    
    # Detect Type (Ocean vs Atmos)
    is_ocean = sum(s in OCEAN_FIELDS for s in suffixes) >= sum(s in ATMOS_FIELDS for s in suffixes)
    ref_list = OCEAN_FIELDS if is_ocean else ATMOS_FIELDS
    
    # Sort Variables based on reference list order
    obs_vars = sorted(
        raw_vars, 
        key=lambda v: ref_list.index(v[prefix_len:]) if v[prefix_len:] in ref_list else 999
    )
    
    return obs_vars, is_ocean


def sort_variable_names_by_reference(variable_names, metric):
    """
    Sort a list of variable names (with metric prefix) by standard ocean/atmos order.
    Use this when variables come from multiple datasets that may not all have the same set.

    Args:
        variable_names: List of full variable names (e.g. ["crps_ens.sst", "crps_ens.sit"]).
        metric: Metric prefix (e.g., 'crps_ens').

    Returns:
        tuple: (sorted_list, is_ocean)
            - sorted_list: Sorted list of variable names
            - is_ocean: True if ordered by OCEAN_FIELDS, else ATMOS_FIELDS
    """
    if not variable_names:
        return [], True

    prefix_len = len(metric) + 1
    suffixes = [v[prefix_len:] for v in variable_names if v.startswith(f"{metric}.")]
    if not suffixes:
        return [], True

    is_ocean = sum(s in OCEAN_FIELDS for s in suffixes) >= sum(s in ATMOS_FIELDS for s in suffixes)
    ref_list = OCEAN_FIELDS if is_ocean else ATMOS_FIELDS

    sorted_vars = sorted(
        variable_names,
        key=lambda v: ref_list.index(v[prefix_len:]) if v[prefix_len:] in ref_list else 999,
    )
    return sorted_vars, is_ocean

