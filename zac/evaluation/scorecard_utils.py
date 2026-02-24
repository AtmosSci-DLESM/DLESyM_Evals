"""
scorecard_utils.py

A utility module for generating "Scorecard" heatmaps for weather and climate model evaluation.
This module supports both absolute metric visualization and relative performance comparisons 
against a baseline model (e.g., IFS, Climatology, or a previous experiment).

Features:
    - **Dual Subplots**: Plots performance for two distinct regions (e.g., Global vs Tropics) side-by-side.
    - **Flexible Calculation Modes**:
        - `absolute`: Plots raw metric values (averaged over initialization time).
        - `growth`: Plots the percentage growth of error relative to Lead Time 0.
        - `improvement`: Plots percentage improvement over a baseline (Positive % = Better).
        - `diff`: Plots raw percentage difference ((Exp - Base)/Base).
    - **Automatic Dimension Handling**: Handles 3D input (Init x Lead x Region) by automatically 
      averaging over initialization times.
    - **Visual Styling**: Uses seaborn heatmaps with annotated cells, distinct grid lines, 
      and diverging colormaps for relative comparisons.
    - **Flexible Time Resolution**: Supports baselines with different time resolutions than the 
      experiment (e.g., daily climatology vs 4-day forecast). Baseline values are automatically 
      aligned to the experiment's lead times using nearest-neighbor selection.

Usage Examples:
    >>> import xarray as xr
    >>> from scorecard_utils import plot_scorecard

    # Load your datasets
    >>> ds_exp = xr.open_dataset("experiment_metrics.nc")
    >>> ds_base = xr.open_dataset("baseline_metrics.nc")

    # 1. Absolute Scorecard (Raw values)
    >>> plot_scorecard(ds_exp, metric='crps_ens', regions=['global', 'arctic'], plot_type='absolute')

    # 2. Improvement Scorecard (How much better is Exp vs Base?)
    #    Blue indicates improvement (Exp < Base for error metrics)
    >>> plot_scorecard(ds_exp, metric='crps_ens', ds_baseline=ds_base, regions=['global', 'tropics'], plot_type='improvement')

    # 3. Error Growth Scorecard (How fast does error grow relative to Day 0?)
    >>> plot_scorecard(ds_exp, metric='rmse', regions=['global', 'antarctic'], plot_type='growth')
"""

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
import pandas as pd
import argparse
import yaml
from plotting_utils import load_models_from_config, detect_and_sort_variables, load_baseline

# Standard gravity [m/s^2]; divide geopotential (m^2/s^2) by this to get height in [m] (match score_s2s pipeline)
GEOPOTENTIAL_TO_METERS = 9.80665

def _is_geopotential_var(data_var_name: str, metric: str) -> bool:
    """True if the physical variable is geopotential (z250, z500, z1000, etc.)."""
    prefix = metric + "."
    if not data_var_name.startswith(prefix):
        return False
    physical_var = data_var_name[len(prefix):].strip().lower()
    return physical_var.startswith("z")


def plot_scorecard(
    ds_exp,
    metric,
    ds_baseline=None,
    regions=['global', 'tropics'],
    plot_type='auto',
    output_path=None,
    max_leadtime_days=None,
    cbar_vmin=None,
    cbar_vmax=None,
):
    """
    Plots a scorecard heatmap with flexible calculation modes.
    
    Args:
        ds_exp (xr.Dataset): Experiment dataset.
        metric (str): Metric prefix (e.g., 'crps_ens', 'crps_skill').
        ds_baseline (xr.Dataset, optional): Baseline dataset.
        regions (list): List of exactly two regions to plot.
        plot_type (str): 
            - 'auto': Defaults to 'absolute' (single) or 'improvement' (comparison).
            - 'absolute': Raw values.
            - 'growth': % change relative to the first lead time (Lead 0).
            - 'improvement': % improvement over baseline (Positive = Better).
            - 'diff': Simple % difference ((Exp - Base)/Base).
        output_path (str, optional): Path to save the figure.
        max_leadtime_days (int or float, optional): If set, show only the first N days of lead time (e.g. 45 for 45 days).
        cbar_vmin (float, optional): Explicit lower limit for the colorbar. If None, use plot-type defaults.
        cbar_vmax (float, optional): Explicit upper limit for the colorbar. If None, use plot-type defaults.
    """
    
    # 0. Optional: restrict to first N days of lead time
    if max_leadtime_days is not None:
        if np.issubdtype(ds_exp.lead_time.dtype, np.timedelta64):
            mask = ds_exp.lead_time.dt.days <= max_leadtime_days
        else:
            mask = ds_exp.lead_time.values <= max_leadtime_days
        ds_exp = ds_exp.sel(lead_time=ds_exp.lead_time[mask])
        if ds_baseline is not None:
            if np.issubdtype(ds_baseline.lead_time.dtype, np.timedelta64):
                mask_b = ds_baseline.lead_time.dt.days <= max_leadtime_days
            else:
                mask_b = ds_baseline.lead_time.values <= max_leadtime_days
            ds_baseline = ds_baseline.sel(lead_time=ds_baseline.lead_time[mask_b])
    
    # 1. Determine Plot Mode
    if plot_type == 'auto':
        plot_type = 'improvement' if ds_baseline is not None else 'absolute'
        
    # Validate Mode
    if plot_type in ['improvement', 'diff'] and ds_baseline is None:
        raise ValueError(f"plot_type='{plot_type}' requires ds_baseline.")

    # 2. Identify Variables
    obs_vars, is_ocean = detect_and_sort_variables(ds_exp, metric)
    
    # Create Labels
    prefix_len = len(metric) + 1
    row_labels = [v[prefix_len:].upper() for v in obs_vars]

    # 3. Configure Visuals (Colormap & Labels)
    if plot_type == 'absolute':
        cmap = "Reds"
        center = None
        cbar_label = f"Absolute {metric.upper()}"
        fmt = ".0f"
        vmin, vmax = None, None
        
    elif plot_type == 'growth':
        # Growth usually implies error increase -> Reds
        # If tracking skill decay -> Blues
        cmap = "Reds" 
        center = None
        cbar_label = f"% Change from Lead 0"
        fmt = ".0f"
        vmin, vmax = 0, 100  # Adjust based on how fast error grows
        
    elif plot_type == 'improvement':
        # Positive = Better (Blue), Negative = Worse (Red)
        # Note: This assumes 'Error' metrics (CRPS/RMSE) where lowering the value is good.
        # If plotting Skill (ACC), you might want to swap this colorbar.
        cmap = "RdBu_r" 
        center = 0
        cbar_label = f"% Improvement vs Baseline (Blue = Better)"
        fmt = ".0f"
        vmin, vmax = -100, 100
        
    elif plot_type == 'diff':
        # Raw Diff: (Exp - Base) / Base
        # For error metrics: Negative = Better (Blue)
        cmap = "RdBu_r"
        center = 0
        cbar_label = f"% Difference (Exp - Base) / Base"
        fmt = ".0f"
        vmin, vmax = -100, 100

    # Allow explicit overrides from config/CLI while keeping sensible defaults above
    if (cbar_vmin is not None) or (cbar_vmax is not None):
        if cbar_vmin is not None:
            vmin = cbar_vmin
        if cbar_vmax is not None:
            vmax = cbar_vmax

    # 4. Setup Figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), constrained_layout=True)
    
    for i, region in enumerate(regions):
        ax = axes[i]
        
        # Build Data Matrix
        data_matrix = []
        for v in obs_vars:
            # Aggregate over init_time and any ensemble dimensions (number) first
            exp_data = ds_exp[v].sel(region=region)
            # Average over init_time if present
            if 'init_time' in exp_data.dims:
                exp_data = exp_data.mean(dim="init_time", skipna=True)
            # Average over ensemble members (number) if present
            if 'number' in exp_data.dims:
                exp_data = exp_data.mean(dim="number", skipna=True)
            val_exp = exp_data.values
            # Convert geopotential to height [m] for comparability with score_s2s pipeline
            if _is_geopotential_var(v, metric):
                val_exp = val_exp / GEOPOTENTIAL_TO_METERS
            
            # --- Calculation Logic ---
            if plot_type == 'absolute':
                val_final = val_exp
                
            elif plot_type == 'growth':
                # (Val_t - Val_0) / Val_0 * 100
                initial_val = val_exp[0] 
                # Avoid divide by zero
                initial_val = np.where(initial_val == 0, np.nan, initial_val)
                val_final = ((val_exp - initial_val) / initial_val) * 100
                
            elif plot_type == 'improvement':
                # (Base - Exp) / Base * 100
                # Positive result means Exp is lower (better) than Base
                # Align baseline to experiment's lead times (handles different time resolutions)
                baseline_data = ds_baseline[v].sel(region=region)
                if 'init_time' in baseline_data.dims:
                    baseline_data = baseline_data.mean(dim="init_time", skipna=True)
                if 'number' in baseline_data.dims:
                    baseline_data = baseline_data.mean(dim="number", skipna=True)
                # Select baseline at experiment's lead times using nearest-neighbor matching
                exp_lead_times = ds_exp.lead_time.values
                val_base = baseline_data.sel(lead_time=exp_lead_times, method='nearest').values
                if _is_geopotential_var(v, metric):
                    val_base = val_base / GEOPOTENTIAL_TO_METERS
                val_final = ((val_base - val_exp) / val_base) * 100
                
            elif plot_type == 'diff':
                # (Exp - Base) / Base * 100
                # Negative result means Exp is lower (better) than Base
                # Align baseline to experiment's lead times (handles different time resolutions)
                baseline_data = ds_baseline[v].sel(region=region)
                if 'init_time' in baseline_data.dims:
                    baseline_data = baseline_data.mean(dim="init_time", skipna=True)
                if 'number' in baseline_data.dims:
                    baseline_data = baseline_data.mean(dim="number", skipna=True)
                # Select baseline at experiment's lead times using nearest-neighbor matching
                exp_lead_times = ds_exp.lead_time.values
                val_base = baseline_data.sel(lead_time=exp_lead_times, method='nearest').values
                if _is_geopotential_var(v, metric):
                    val_base = val_base / GEOPOTENTIAL_TO_METERS
                val_final = ((val_exp - val_base) / val_base) * 100
                
            data_matrix.append(val_final)
        
        data_matrix = np.array(data_matrix)
        
        # DataFrame for Plotting
        lead_times = ds_exp.lead_time.dt.days.values
        df = pd.DataFrame(data_matrix, index=row_labels, columns=lead_times)

        # Heatmap (small font in cells to avoid overlap; values shown as integers)
        sns.heatmap(
            df, 
            ax=ax,
            annot=True, 
            fmt=fmt, 
            cmap=cmap, 
            center=center,
            vmin=vmin, vmax=vmax,
            linewidths=1, 
            linecolor='white',
            annot_kws={'fontsize': 7},
            cbar_kws={'label': cbar_label if i == 1 else None}
        )
        
        # Formatting
        ax.set_title(f"{plot_type.capitalize().replace('_', ' ')}: {region.upper()}", 
                     fontsize=14, loc='left', fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        if i == len(regions) - 1:
            ax.set_xlabel("Lead time [days]", fontsize=12)
        else:
            ax.set_xlabel("")

    fig.suptitle(f"{metric} Scorecard - {plot_type.capitalize()}", fontsize=16, y=1.05)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()




def main():
    """CLI entry point for scorecard_utils.py"""
    parser = argparse.ArgumentParser(description="Generate scorecard heatmaps from evaluation results")
    parser.add_argument("--config", "-c", help="Path to plotting config YAML file (for multi-model comparison)")
    parser.add_argument("--input", "-i", help="Path to evaluation timeseries NetCDF file (single model mode)")
    parser.add_argument("--output", "-o", required=True, help="Path to save output figure")
    parser.add_argument("--metric", "-m", default="crps_ens", help="Metric prefix to plot (default: crps_ens)")
    parser.add_argument("--plot-type", "-p", default="auto", choices=["absolute", "growth", "improvement", "diff", "auto"],
                       help="Plot type (default: auto)")
    parser.add_argument("--regions", "-r", nargs=2, default=["global", "arctic"],
                       help="Two regions to plot (default: global arctic)")
    parser.add_argument("--baseline", "-b", help="Path to baseline dataset for comparison plots (can be climatology, persistence, or model)")
    parser.add_argument("--baseline-label", help="Label for baseline (default: auto-detect)")
    parser.add_argument("--max-leadtime-days", type=float, default=None,
                        help="Show only the first N days of lead time (e.g. 45 for 45 days). Overrides config.")
    
    args = parser.parse_args()
    
    # Determine mode: config-based (multi-model) or single model
    if args.config:
        # Multi-model mode: load from config
        if args.input:
            print("WARNING: Both --config and --input provided. Using --config (multi-model mode).")
        
        experiments, config_baseline_label, config_baseline_ds = load_models_from_config(args.config)
        
        # Override metric and regions from config if available
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        plotting_cfg = config.get('plotting', {})
        metric = plotting_cfg.get('metric', args.metric)
        regions = plotting_cfg.get('regions', args.regions)
        max_leadtime_days = args.max_leadtime_days if args.max_leadtime_days is not None else plotting_cfg.get('max_leadtime_days')

        # Optional colorbar limits from config: plotting.colorbar_limits: [vmin, vmax]
        colorbar_limits = plotting_cfg.get('colorbar_limits')
        cbar_vmin = cbar_vmax = None
        if isinstance(colorbar_limits, (list, tuple)) and len(colorbar_limits) == 2:
            cbar_vmin, cbar_vmax = colorbar_limits
        
        # Determine baseline: command line takes precedence, then model baseline, then config baseline
        if args.baseline:
            # Command line baseline takes highest priority
            baseline_ds, baseline_label = load_baseline(
                baseline_path=args.baseline,
                baseline_label=args.baseline_label
            )
        elif config_baseline_ds is not None:
            # Use model baseline from config
            baseline_ds = config_baseline_ds
            baseline_label = args.baseline_label if args.baseline_label else config_baseline_label
        else:
            # Try loading from baseline config (climatology/persistence)
            baseline_ds, baseline_label = load_baseline(
                baseline_label=args.baseline_label,
                config=config
            )
        
        # Determine plot type based on whether baseline exists
        if baseline_ds is not None:
            plot_type = 'diff'  # Show relative difference vs baseline
        else:
            plot_type = 'absolute'
        
        # Override with command line if provided
        if args.plot_type != 'auto':
            plot_type = args.plot_type
        
        # For scorecard, plot the first non-baseline model (current experiment) vs baseline
        if baseline_ds is None:
            print("WARNING: No baseline specified in config. Plotting absolute values for first model.")
            ds_exp = next(iter(experiments.values()))
            plot_scorecard(
                ds_exp,
                metric=metric,
                ds_baseline=None,
                regions=regions,
                plot_type='absolute',
                output_path=args.output,
                max_leadtime_days=max_leadtime_days,
                cbar_vmin=cbar_vmin,
                cbar_vmax=cbar_vmax,
            )
        else:
            # Find the first model that's not the baseline (current experiment)
            current_model = None
            for label, ds_exp in experiments.items():
                if label != baseline_label:
                    current_model = (label, ds_exp)
                    break
            
            if current_model is None:
                print("WARNING: No non-baseline models found. Plotting baseline as absolute.")
                plot_scorecard(
                    baseline_ds,
                    metric=metric,
                    ds_baseline=None,
                    regions=regions,
                    plot_type='absolute',
                    output_path=args.output,
                    max_leadtime_days=max_leadtime_days,
                    cbar_vmin=cbar_vmin,
                    cbar_vmax=cbar_vmax,
                )
            else:
                # Plot current model vs baseline
                label, ds_exp = current_model
                print(f"Generating scorecard for {label} vs {baseline_label}...")
                plot_scorecard(
                    ds_exp,
                    metric=metric,
                    ds_baseline=baseline_ds,
                    regions=regions,
                    plot_type=plot_type,
                    output_path=args.output,
                    max_leadtime_days=max_leadtime_days,
                    cbar_vmin=cbar_vmin,
                    cbar_vmax=cbar_vmax,
                )
    elif args.input:
        # Single model mode: load single file
        ds_exp = xr.open_dataset(args.input)
        # Load baseline if specified
        ds_baseline, _ = load_baseline(
            baseline_path=args.baseline,
            baseline_label=args.baseline_label
        )
        
        plot_type = args.plot_type
        if plot_type == 'auto':
            plot_type = 'improvement' if ds_baseline is not None else 'absolute'
        
        max_leadtime_days = args.max_leadtime_days
        plot_scorecard(
            ds_exp, 
            metric=args.metric, 
            ds_baseline=ds_baseline,
            regions=args.regions,
            plot_type=plot_type,
            output_path=args.output,
            max_leadtime_days=max_leadtime_days
        )
    else:
        parser.error("Either --config or --input must be provided")


if __name__ == "__main__":
    main()