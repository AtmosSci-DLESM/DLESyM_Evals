"""
DLESyM Data Loaders for WeatherBench-X Evaluation.

This module provides a factory function `get_data_loader()` that wraps WeatherBench-X
built-in data loaders with custom preprocessing functions for DLESyM data format.

Available Loaders:
------------------
  - ClimatologyFromXarray: Deterministic climatology (dayofyear lookup)
  - ProbabilisticClimatologyFromXarray: Ensemble climatology (each year = 1 member)
  - PersistenceFromXarray: Persistence baseline (value at init_time for all lead_times)
  - TargetsFromXarray: Target/analysis datasets (valid_time dimension)
  - DLESyMPredictionsFromXarray: Custom loader for DLESyM prediction format

Preprocessing Functions:
------------------------
  - preprocess_dlesym_targets: Renames dims, unpacks channels, drops duplicates
  - preprocess_dlesym_climatology: Renames dims, unpacks channels for climatology
  - preprocess_dlesym_predictions: Renames dims for prediction datasets

For details on WBX loaders, see:
https://weatherbench-x.readthedocs.io/en/latest/api/data_loaders.html

Land-sea masking: When evaluation config sets align_lsm_with_infill: true, the target
loader receives a masking_cfg built from the same LSM path and land_threshold as
training/inference ocean_land_infill, so infilled values do not contribute to skill.
When false or omitted, behavior is unchanged (backward compatible).
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import xarray as xr
import zarr

from weatherbenchX.data_loaders import base as data_loader_base
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX import interpolations

# Setup logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Open DLESyM zarr (decode_cf=False so _FillValue=0 does not turn 0s into NaN)
# ------------------------------------------------------------------------------

def open_dlesym_zarr(path: str) -> xr.Dataset:
    """
    Open a .zarr store with decode_cf=False and drop _FillValue/missing_value attrs.
    Use this so the dataset can be passed to WBX loaders via ds= and 0s stay 0.
    """
    consolidate_zarr_metadata(path)
    try:
        ds = xr.open_zarr(path, consolidated=True, decode_cf=False)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False, decode_cf=False)
    for v in ds.data_vars:
        ds[v].attrs.pop("_FillValue", None)
        ds[v].attrs.pop("missing_value", None)
    return ds


def consolidate_zarr_metadata(path: str) -> bool:
    """Attempt to consolidate zarr metadata for faster opening."""
    try:
        zarr.open_group(path, mode="r")
        zarr.consolidate_metadata(path)
        return True
    except Exception as e:
        logger.debug("Could not consolidate zarr metadata: %s", e)
        return False


# ------------------------------------------------------------------------------
# Preprocessing Functions for WBX Loaders
# ------------------------------------------------------------------------------

def ensure_latitude_ascending(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude coordinate is monotonically increasing (WBX expectation)."""
    if "latitude" in ds.coords:
        lat = ds["latitude"]
        if lat.isnull().any():
            ds = ds.sel(latitude=~lat.isnull())
            lat = ds["latitude"]
        if lat.size > 1 and float(lat[0]) > float(lat[-1]):
            ds = ds.sortby("latitude")
    return ds


def fill_prediction_nans(ds: xr.Dataset, loader_name: str) -> xr.Dataset:
    """
    Fill NaN values with 0 in all data variables (for prediction loaders).
    Logs when any NaNs are replaced.
    """
    any_filled = False
    for var in list(ds.data_vars):
        da = ds[var]
        if da.isnull().any():
            # .compute() needed when array is dask-backed (dask has no .item())
            n_nan = int(da.isnull().sum().compute().item())
            ds[var] = da.fillna(0)
            logger.info(f"[{loader_name}] Filled {n_nan} NaN(s) with 0 in variable '{var}'")
            any_filled = True
    if any_filled:
        logger.info(f"[{loader_name}] NaN filling applied to prediction data.")
    return ds


def load_land_sea_mask(masking_cfg: dict) -> xr.DataArray:
    """
    Load land-sea mask from config and return a boolean mask.
    When align_lsm_with_infill is true in the evaluation config, the caller
    builds masking_cfg from the same LSM path and land_threshold as
    training/inference infilling (backward compatible when false or omitted).
    Args:
        masking_cfg: Masking configuration dict with keys:
            - enabled: bool
            - path: str (path to mask file)
            - variable: str (variable name in file, e.g., 'lsm')
            - threshold: float (values >= threshold are land)
            - type: str ('land' to mask land, 'ocean' to mask ocean)
    
    Returns:
        xr.DataArray: Boolean mask where True = keep, False = mask out
        Returns None if masking is disabled or config is invalid
    """
    if not masking_cfg or not masking_cfg.get('enabled', False):
        return None
    
    mask_path = masking_cfg.get('path')
    mask_var = masking_cfg.get('variable', 'lsm')
    threshold = masking_cfg.get('threshold', 0.05)
    mask_type = masking_cfg.get('type', 'land')  # 'land' = mask out land (keep ocean)
    
    if not mask_path:
        logger.warning("Masking enabled but no path specified. Skipping masking.")
        return None
    
    try:
        logger.info(f"Loading land-sea mask from: {mask_path}")
        if mask_path.endswith('.zarr'):
            mask_ds = xr.open_zarr(mask_path)
        else:
            mask_ds = xr.open_dataset(mask_path)
        
        # Get the mask variable
        if mask_var not in mask_ds:
            available = list(mask_ds.data_vars)
            logger.warning(f"Mask variable '{mask_var}' not found. Available: {available}")
            return None
        
        lsm = mask_ds[mask_var]
        
        # Rename coords to match target data convention
        rename_map = {}
        if 'lat' in lsm.coords:
            rename_map['lat'] = 'latitude'
        if 'lon' in lsm.coords:
            rename_map['lon'] = 'longitude'
        if rename_map:
            lsm = lsm.rename(rename_map)
        
        # Remove time dimension if present (take first timestep)
        if 'time' in lsm.dims:
            lsm = lsm.isel(time=0)
        if 'valid_time' in lsm.dims:
            lsm = lsm.isel(valid_time=0)
        
        # Create boolean mask: True = keep, False = mask out
        # lsm >= threshold means it's land
        is_land = lsm >= threshold
        
        if mask_type == 'land':
            # Keep ocean (mask out land) -> keep where NOT land
            keep_mask = ~is_land
            logger.info(f"Land-sea mask loaded. Masking out LAND (keeping ocean). Threshold: {threshold}")
        else:
            # Keep land (mask out ocean) -> keep where IS land
            keep_mask = is_land
            logger.info(f"Land-sea mask loaded. Masking out OCEAN (keeping land). Threshold: {threshold}")
        
        return keep_mask.load()  # Load into memory for efficiency
        
    except Exception as e:
        logger.error(f"Failed to load land-sea mask: {e}")
        return None


def preprocess_dlesym_targets(ds: xr.Dataset, masking_cfg: dict = None) -> xr.Dataset:
    """
    Preprocessing for DLESyM target datasets.
    Renames lat/lon and time, drops duplicate timestamps, unpacks targets/constants,
    applies land-sea mask if configured. (Call open_dlesym_zarr first so 0s stay 0.)
    """
    # 1. Rename dims/coords to WeatherBenchX convention first
    # This standardizes the names so we only have to clean 'valid_time'
    rename_map = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if "time" in ds.dims or "time" in ds.coords:
        rename_map["time"] = "valid_time"
    
    # Apply renames if they exist in the dataset
    dataset_rename = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    if dataset_rename:
        ds = ds.rename(dataset_rename)

    # Restore indexes after rename (xarray rename often drops indexes)
    for new_name in dataset_rename.values():
        if new_name in ds.coords and new_name not in ds.indexes:
            # Only set index if the dimension exists
            if new_name in ds.dims:
                ds = ds.set_xindex(new_name)

    # 2. Drop duplicates using Xarray native method (The Critical Fix)
    # This ensures the index is strictly unique for .sel() operations later
    if "valid_time" in ds.dims:
        original_size = ds.sizes["valid_time"]
        ds = ds.drop_duplicates(dim="valid_time")

        # --- ADD THIS LINE ---
        # Force the valid_time index into memory immediately to ensure uniqueness is finalized
        if "valid_time" in ds.coords:
            ds["valid_time"] = ds["valid_time"].compute()
        # ---------------------
        
        new_size = ds.sizes["valid_time"]
        if original_size != new_size:
            logger.warning(f"Dropped {original_size - new_size} duplicate timestamps from 'valid_time'")

    ds = ensure_latitude_ascending(ds)

    # 3. Unpack targets array by channel_out
    if "targets" in ds and "channel_out" in ds["targets"].coords:
        for ch in ds["channel_out"].values:
            vname = str(ch)
            ds[vname] = ds["targets"].sel(channel_out=ch).drop_vars("channel_out")
        ds = ds.drop_vars("targets")
    
    # 4. Unpack constants array by channel_c
    if "constants" in ds and "channel_c" in ds["constants"].coords:
        for ch in ds["channel_c"].values:
            vname = str(ch)
            ds[vname] = ds["constants"].sel(channel_c=ch).drop_vars("channel_c")
        ds = ds.drop_vars("constants")
    
    # Drop inputs if present
    if "inputs" in ds:
        ds = ds.drop_vars("inputs")
    
    # Drop channel dimensions if present
    for dim in ["channel_out", "channel_c", "channel_in"]:
        if dim in ds.dims:
            ds = ds.drop_dims(dim)
    
    # Apply land-sea mask if configured (sets masked points to NaN)
    if masking_cfg:
        land_sea_mask = load_land_sea_mask(masking_cfg)
        if land_sea_mask is not None:
            # Align mask to dataset coordinates
            mask_aligned = land_sea_mask.reindex_like(ds, method='nearest')
            # Apply mask to all data variables (sets masked points to NaN)
            for var in ds.data_vars:
                ds[var] = ds[var].where(mask_aligned)
            logger.info("Applied land-sea mask to target dataset")
    
    return ds


def preprocess_dlesym_climatology(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocessing for DLESyM climatology. Renames lat/lon, unpacks targets by channel_out.
    (Call open_dlesym_zarr first so 0s stay 0.)
    """
    # Rename spatial dims/coords
    dim_map = {}
    if "lat" in ds.dims:
        dim_map["lat"] = "latitude"
    if "lon" in ds.dims:
        dim_map["lon"] = "longitude"
    if dim_map:
        ds = ds.rename_dims(dim_map)
    
    coord_map = {}
    if "lat" in ds.coords:
        coord_map["lat"] = "latitude"
    if "lon" in ds.coords:
        coord_map["lon"] = "longitude"
    if coord_map:
        ds = ds.rename(coord_map)
    
    ds = ensure_latitude_ascending(ds)
    
    # Unpack targets array by channel_out
    if "targets" in ds and "channel_out" in ds["targets"].coords:
        for ch in ds["channel_out"].values:
            vname = str(ch)
            ds[vname] = ds["targets"].sel(channel_out=ch).drop_vars("channel_out")
        ds = ds.drop_vars("targets")
    
    # Drop channel dimensions if present
    for dim in ["channel_out", "channel_c", "channel_in"]:
        if dim in ds.dims:
            ds = ds.drop_dims(dim)
    
    ds = fill_prediction_nans(ds, "climatology")
    return ds


def preprocess_dlesym_predictions(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocessing for DLESyM predictions. Renames lat/lon, ensures latitude ascending.
    (Call open_dlesym_zarr first so 0s stay 0.)
    """
    # Rename spatial dims/coords
    dim_map = {}
    if "lat" in ds.dims:
        dim_map["lat"] = "latitude"
    if "lon" in ds.dims:
        dim_map["lon"] = "longitude"
    if dim_map:
        ds = ds.rename_dims(dim_map)
    
    coord_map = {}
    if "lat" in ds.coords:
        coord_map["lat"] = "latitude"
    if "lon" in ds.coords:
        coord_map["lon"] = "longitude"
    if coord_map:
        ds = ds.rename(coord_map)
    
    ds = ensure_latitude_ascending(ds)
    ds = fill_prediction_nans(ds, "predictions")
    return ds


# ------------------------------------------------------------------------------
# Custom DLESyM Prediction Loader (for non-standard prediction format)
# ------------------------------------------------------------------------------

class DLESyMPredictionsFromXarray(data_loader_base.DataLoader):
    """
    Custom Data Loader for DLESyM Predictions.
    Handles the packed zarr format with init_time and lead_time dimensions.
    """
    def __init__(
        self, 
        path: str, 
        variables: Optional[List[str]] = None, 
        rename: Optional[Dict] = None,
        interpolation: Optional[interpolations.Interpolation] = None,
        compute: bool = True,
        add_nan_mask: bool = False,
    ):
        super().__init__(
            interpolation=interpolation,
            compute=compute,
            add_nan_mask=False,
        )
        self.path = path
        
        logger.info(f"Initializing Prediction Loader. Opening files from: {path}")
        ds = open_dlesym_zarr(path)
        self._ds = preprocess_dlesym_predictions(ds)
        
        if rename:
            self._ds = self._ds.rename(rename)
        
        if variables:
            missing = [v for v in variables if v not in self._ds]
            if missing:
                raise ValueError(f"Variables {missing} not found in prediction dataset.")
            self._ds = self._ds[variables]
        
        # Rename ensemble dimension AND coordinate to 'number' (WBX convention)
        if 'ensemble' in self._ds.dims:
            self._ds = self._ds.rename_dims({"ensemble": "number"})
        if 'ensemble' in self._ds.coords:
            self._ds = self._ds.rename({"ensemble": "number"})

    def _load_chunk_from_source(self, init_times, lead_times):
        """Load a chunk of predictions for given init_times and lead_times."""
        chunk = self._ds.sel(
            init_time=xr.DataArray(init_times, dims="init_time"),
            lead_time=xr.DataArray(lead_times, dims="lead_time")
        )
        return chunk


# ------------------------------------------------------------------------------
# Factory Function with WBX Loaders
# ------------------------------------------------------------------------------

def get_data_loader(config: Dict[str, Any]):
    """
    Factory to initialize data loaders.
    
    Uses WBX built-in loaders where possible with custom preprocessing functions:
    - ClimatologyFromXarray: For deterministic climatology baselines
    - ProbabilisticClimatologyFromXarray: For probabilistic climatology baselines  
    - PersistenceFromXarray: For persistence baselines
    - TargetsFromXarray: For target/analysis datasets
    - DLESyMPredictionsFromXarray: Custom loader for DLESyM predictions
    """
    cls_name = config.get("class")
    path = config["path"]
    variables = config.get("variables")
    rename = config.get("rename")
    compute = config.get("compute", True)
    # Default add_nan_mask behavior:
    # - For most loaders, default to False (backwards compatible).
    # - For target loaders, we override this to default to True while still
    #   respecting any explicit add_nan_mask value in the config.
    add_nan_mask = config.get("add_nan_mask", False)

    # --------------------------------------------------
    # WBX ClimatologyFromXarray (deterministic climatology)
    # --------------------------------------------------
    if cls_name == "ClimatologyFromXarray":
        ds = open_dlesym_zarr(path)
        climatology_time_coords = ("dayofyear", "hour") if "hour" in ds.dims else ("dayofyear",)
        loader = xarray_loaders.ClimatologyFromXarray(
            ds=ds,
            variables=variables,
            rename_variables=rename,
            climatology_time_coords=climatology_time_coords,
            rename_dimensions=None,
            preprocessing_fn=preprocess_dlesym_climatology,
            compute=compute,
            add_nan_mask=add_nan_mask,
        )
        logger.info("Using WBX ClimatologyFromXarray. Variables: %s", variables)
        logger.info("Climatology time coords: %s", climatology_time_coords)
        return loader
    
    # --------------------------------------------------
    # WBX ProbabilisticClimatologyFromXarray (ensemble climatology)
    # --------------------------------------------------
    if cls_name == "ProbabilisticClimatologyFromXarray":
        start_year = config.get("start_year", 2000)
        end_year = config.get("end_year", 2020)

        def preprocess_prob_climatology(ds_in):
            ds_in = preprocess_dlesym_targets(ds_in)
            return fill_prediction_nans(ds_in, "probabilistic_climatology")

        ds = open_dlesym_zarr(path)
        loader = xarray_loaders.ProbabilisticClimatologyFromXarray(
            ds=ds,
            variables=variables,
            rename_variables=rename,
            start_year=start_year,
            end_year=end_year,
            ensemble_dim="number",
            rename_dimensions=None,
            preprocessing_fn=preprocess_prob_climatology,
            compute=compute,
            add_nan_mask=add_nan_mask,
        )
        logger.info("Using WBX ProbabilisticClimatologyFromXarray. Years: %s-%s", start_year, end_year)
        return loader
    
    # --------------------------------------------------
    # WBX PersistenceFromXarray
    # --------------------------------------------------
    if cls_name == "PersistenceFromXarray":
        def preprocess_persistence(ds_in):
            ds_in = preprocess_dlesym_targets(ds_in)
            return fill_prediction_nans(ds_in, "persistence")

        ds = open_dlesym_zarr(path)
        loader = xarray_loaders.PersistenceFromXarray(
            ds=ds,
            variables=variables,
            rename_variables=rename,
            rename_dimensions=None,
            preprocessing_fn=preprocess_persistence,
            compute=compute,
            add_nan_mask=add_nan_mask,
        )
        logger.info("Using WBX PersistenceFromXarray.")
        return loader
    
    # --------------------------------------------------
    # WBX TargetsFromXarray
    # --------------------------------------------------
    if cls_name == "TargetsFromXarray" or cls_name == "DLESyMTargetsFromXarray":
        masking_cfg = config.get("masking")
        target_add_nan_mask = config.get("add_nan_mask", True)

        def preprocess_with_mask(ds_in):
            return preprocess_dlesym_targets(ds_in, masking_cfg=masking_cfg)

        ds = open_dlesym_zarr(path)
        loader = xarray_loaders.TargetsFromXarray(
            ds=ds,
            variables=variables,
            rename_variables=rename,
            rename_dimensions=None,
            preprocessing_fn=preprocess_with_mask,
            compute=compute,
            add_nan_mask=target_add_nan_mask,
        )
        logger.info("Using WBX TargetsFromXarray. add_nan_mask=%s", target_add_nan_mask)
        return loader
    
    # --------------------------------------------------
    # Custom DLESyM Predictions Loader
    # --------------------------------------------------
    if cls_name == "DLESyMPredictionsFromXarray":
        loader = DLESyMPredictionsFromXarray(
            path=path,
            variables=variables,
            rename=rename,
            compute=compute,
            add_nan_mask=add_nan_mask,
        )
        return loader
    
    raise ValueError(
        f"Data loader class '{cls_name}' not found.\n"
        f"Available loaders:\n"
        f"  - ClimatologyFromXarray: Deterministic climatology (dayofyear lookup)\n"
        f"  - ProbabilisticClimatologyFromXarray: Ensemble climatology (each year = 1 member)\n"
        f"  - PersistenceFromXarray: Persistence baseline (value at init_time)\n"
        f"  - TargetsFromXarray: Target/analysis datasets\n"
        f"  - DLESyMPredictionsFromXarray: DLESyM prediction format"
    )


# ------------------------------------------------------------------------------
# Climatology Processing (for metrics that need climatology dataset)
# ------------------------------------------------------------------------------

def process_climatology_dataset(clim_path: str, variables: Optional[List[str]] = None) -> xr.Dataset:
    """
    Process climatology dataset to match WeatherBench-X format.
    Used for ACC metric climatology reference.
    """
    logger.info("Loading climatology from: %s", clim_path)
    ds = open_dlesym_zarr(clim_path)
    ds = preprocess_dlesym_climatology(ds)

    if variables:
        available = [v for v in variables if v in ds]
        if not available:
            logger.warning(f"No requested variables found in climatology. Available: {list(ds.data_vars)}")
        ds = ds[available]
    
    logger.info(f"Climatology processed. Variables: {list(ds.data_vars)}")
    logger.info(f"Climatology dimensions: {dict(ds.dims)}")
    
    return ds
