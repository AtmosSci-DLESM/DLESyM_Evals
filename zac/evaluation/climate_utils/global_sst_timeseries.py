"""
global_sst_timeseries.py

Analysis function for plotting global average SST timeseries from climate forecasts.

This function computes a globally-averaged SST timeseries (weighted by latitude)
and optionally creates a plot. It handles various time dimension formats commonly
found in climate forecast datasets.
"""

import logging
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def plot_global_sst_timeseries(
    inference_file: xr.Dataset,
    target_file: xr.Dataset,
    variable: str = "sst",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> Optional[xr.DataArray]:
    """
    Compute and plot global average SST timeseries for both model and observations.
    
    Computes a latitude-weighted global average of the specified variable
    over time for both the inference (model) and target (observations) datasets.
    Plots both timeseries for comparison.
    
    Args:
        inference_file: Preprocessed climate inference dataset (model predictions)
        target_file: Preprocessed target dataset (observations/reference)
        variable: Variable name to analyze (default: "sst")
        output_path: Path to save plot (optional). If None, plot is not saved.
        title: Plot title (optional). If None, auto-generated from variable name.
        **kwargs: Additional arguments (ignored)
    
    Returns:
        xarray DataArray with model global average timeseries, or None if error
    
    Raises:
        ValueError: If variable is not found in dataset
        KeyError: If required coordinates (latitude, longitude) are missing
    """
    # Check if variable exists in both datasets
    if variable not in inference_file.data_vars:
        available = list(inference_file.data_vars)
        logger.error(f"Variable '{variable}' not found in inference dataset. Available: {available}")
        raise ValueError(f"Variable '{variable}' not found in inference dataset")
    
    if variable not in target_file.data_vars:
        available = list(target_file.data_vars)
        logger.warning(f"Variable '{variable}' not found in target dataset. Available: {available}")
        logger.warning("Will plot model timeseries only (no comparison)")
        use_target = False
    else:
        use_target = True
    
    logger.info(f"Computing global average timeseries for variable: {variable}")
    
    # Get the variables
    model_data = inference_file[variable]
    if use_target:
        target_data = target_file[variable]
    
    # Helper function to compute global average
    def compute_global_avg(data_var: xr.DataArray) -> Tuple[xr.DataArray, Optional[str]]:
        """Compute latitude-weighted global average and return timeseries and time dimension name."""
        # Check for required coordinates
        if "latitude" not in data_var.coords:
            logger.error("Dataset missing 'latitude' coordinate")
            raise KeyError("Dataset missing 'latitude' coordinate")
        if "longitude" not in data_var.coords:
            logger.error("Dataset missing 'longitude' coordinate")
            raise KeyError("Dataset missing 'longitude' coordinate")
        
        # Find time dimension (could be time, valid_time, or lead_time)
        time_dims = ["time", "valid_time", "lead_time"]
        time_dim = None
        for dim in time_dims:
            if dim in data_var.dims:
                time_dim = dim
                break
        
        if time_dim is None:
            logger.warning("No time dimension found. Computing spatial average only.")
            time_dim = None
        else:
            logger.debug(f"Using time dimension: {time_dim}")
        
        # Compute latitude weights (cosine of latitude for area weighting)
        lat = data_var["latitude"]
        # Convert to radians and compute cosine
        lat_rad = np.deg2rad(lat.values)
        cos_lat = np.cos(lat_rad)
        # Normalize weights
        weights = cos_lat / cos_lat.sum()
        
        # Create weights DataArray aligned with data
        weights_da = xr.DataArray(
            weights,
            dims=["latitude"],
            coords={"latitude": lat}
        )
        
        # Compute global average (weighted by latitude, mean over longitude)
        # Handle NaN values (e.g., land mask)
        if time_dim:
            # Timeseries: average over lat/lon, keep time dimension
            global_avg = (
                data_var.weighted(weights_da)
                .mean(dim=["latitude", "longitude"], skipna=True)
            )
        else:
            # No time dimension: just spatial average
            global_avg = (
                data_var.weighted(weights_da)
                .mean(dim=["latitude", "longitude"], skipna=True)
            )
        
        return global_avg, time_dim
    
    # Compute global averages for both datasets
    model_avg, model_time_dim = compute_global_avg(model_data)
    if use_target:
        target_avg, target_time_dim = compute_global_avg(target_data)
    
    # Use model's global average as return value
    global_avg = model_avg
    time_dim = model_time_dim
    
    # Convert to DataArray if it's not already
    if not isinstance(global_avg, xr.DataArray):
        global_avg = xr.DataArray(global_avg)
    
    logger.info(f"Computed model global average. Shape: {global_avg.shape}, Mean: {float(global_avg.mean().values):.4f}")
    if use_target:
        logger.info(f"Computed target global average. Shape: {target_avg.shape}, Mean: {float(target_avg.mean().values):.4f}")
    
    # Create plot if output_path is provided
    if output_path is not None:
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot timeseries
            if time_dim:
                # Get time values for x-axis (use model's time)
                model_time_coord = model_avg[time_dim]
                
                # Handle different time formats
                if np.issubdtype(model_time_coord.dtype, np.datetime64):
                    # Datetime - plot directly
                    ax.plot(model_time_coord.values, model_avg.values, linewidth=1.5, label="Model", color="blue")
                    if use_target:
                        # Align target to model time if possible, or use target's own time
                        if target_time_dim and target_time_dim in target_avg.dims:
                            target_time_coord = target_avg[target_time_dim]
                            ax.plot(target_time_coord.values, target_avg.values, linewidth=1.5, label="Observations", color="red", alpha=0.7)
                        else:
                            # Try to align by time values
                            try:
                                # Interpolate target to model time
                                target_aligned = target_avg.interp({target_time_dim: model_time_coord}, method="nearest")
                                ax.plot(model_time_coord.values, target_aligned.values, linewidth=1.5, label="Observations", color="red", alpha=0.7)
                            except Exception:
                                logger.warning("Could not align target timeseries to model time. Plotting model only.")
                    # Format x-axis dates
                    fig.autofmt_xdate()
                else:
                    # Numeric time - plot as is
                    ax.plot(model_time_coord.values, model_avg.values, linewidth=1.5, label="Model", color="blue")
                    if use_target and target_time_dim and target_time_dim in target_avg.dims:
                        target_time_coord = target_avg[target_time_dim]
                        ax.plot(target_time_coord.values, target_avg.values, linewidth=1.5, label="Observations", color="red", alpha=0.7)
                
                ax.set_xlabel("Time", fontsize=12)
                ax.legend(loc="best", fontsize=10)
            else:
                # No time dimension - just a single value
                ax.axhline(y=float(model_avg.values), linewidth=2, label="Model", color="blue")
                if use_target:
                    ax.axhline(y=float(target_avg.values), linewidth=2, label="Observations", color="red", alpha=0.7)
                ax.set_xlabel("Spatial Average", fontsize=12)
                ax.legend(loc="best", fontsize=10)
            
            # Set labels and title
            ax.set_ylabel(f"{variable.upper()} (Global Average)", fontsize=12)
            if title:
                ax.set_title(title, fontsize=14, fontweight="bold")
            else:
                title_text = f"Global Average {variable.upper()} Timeseries"
                if use_target:
                    title_text += " (Model vs Observations)"
                ax.set_title(title_text, fontsize=14, fontweight="bold")
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)
            
            plt.tight_layout()
            
            # Save figure
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved plot to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create plot: {e}", exc_info=True)
            # Don't fail the whole analysis if plotting fails
    
    return global_avg

