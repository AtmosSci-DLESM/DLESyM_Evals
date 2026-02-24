"""
skill_curves.py

A utility module for comparing multiple experiments using line plots. 
It generates a grid of subplots (one per variable) where each line represents a different experiment.

Features:
    - **Smart Sorting**: Automatically detects Ocean vs. Atmosphere variables and sorts 
      them according to standard conventions.
    - **Dynamic Square Grid**: Automatically calculates the best grid dimensions (e.g., 3x3 for 9 vars) 
      to keep the layout compact.
    - **Multi-Experiment Support**: Accepts a dictionary of datasets to plot arbitrary numbers of experiments.
    - **Automatic averaging**: Handles 'init_time' averaging automatically.

Usage Examples:
    >>> import xarray as xr
    >>> from skill_curves import plot_skill_curves

    # Load datasets
    >>> ds_ours = xr.open_dataset("results_ours.nc")
    >>> ds_base = xr.open_dataset("results_baseline.nc")
    
    # Create dictionary of experiments
    >>> experiments = {
    ...     "Our Model": ds_ours,
    ...     "Baseline": ds_baseline,
    ...     "IFS HRES": xr.open_dataset("results_ifs.nc")
    ... }

    # Generate grid for Global region
    >>> plot_skill_curves(experiments, metric='crps_ens', region='global')
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
import argparse
import sys
import os
import yaml
import xarray as xr
from plotting_utils import load_models_from_config, sort_variable_names_by_reference, load_baseline

# Standard gravity [m/s^2]; divide geopotential (m^2/s^2) by this to get height in [m] (match score_s2s pipeline)
GEOPOTENTIAL_TO_METERS = 9.80665

def _is_geopotential_var(data_var_name: str, metric: str) -> bool:
    """True if the physical variable is geopotential (z250, z500, z1000, etc.)."""
    prefix = metric + "."
    if not data_var_name.startswith(prefix):
        return False
    physical_var = data_var_name[len(prefix):].strip().lower()
    return physical_var.startswith("z")


def _plot_rmse_ens_if_present(ax, ds, var_name, region, x_vals_full, mask_lt, max_leadtime_days, color, linestyle):
    """When metric is rmse, plot rmse_ens.<suffix> as dashed line in same color if present in ds."""
    # var_name is e.g. "rmse.sst" -> suffix "sst", rmse_ens_var "rmse_ens.sst"
    suffix = var_name.split(".", 1)[1]
    rmse_ens_var = "rmse_ens." + suffix
    if rmse_ens_var not in ds.data_vars:
        return
    data = ds[rmse_ens_var].sel(region=region)
    if 'init_time' in data.dims:
        data = data.mean(dim="init_time", skipna=True)
    if 'number' in data.dims:
        data = data.mean(dim="number", skipna=True)
    y_vals = data.values.astype(float)
    if _is_geopotential_var(rmse_ens_var, "rmse_ens"):
        y_vals = y_vals / GEOPOTENTIAL_TO_METERS
    if max_leadtime_days is not None:
        x_plot = np.asarray(x_vals_full)[mask_lt]
        y_vals = y_vals[mask_lt]
    else:
        x_plot = x_vals_full
    ax.plot(x_plot, y_vals, color=color, linewidth=2, linestyle=linestyle)


def plot_skill_curves(
    experiments,
    metric,
    region='global',
    baseline_ds=None,
    baseline_label=None,
    output_path=None,
    max_leadtime_days=None,
    show_ensemble_members=False,
):
    """
    Generates a grid of line plots comparing multiple experiments.

    Args:
        experiments (dict): Dictionary where keys are labels (str) and values are xarray Datasets.
        metric (str): The metric prefix to extract (e.g., 'crps_ens').
        region (str): The specific region to plot (e.g., 'global').
        baseline_ds (xr.Dataset, optional): Baseline dataset (climatology or persistence) to plot as dotted line.
        baseline_label (str, optional): Label for baseline (default: "Climatology" or "Persistence").
        output_path (str, optional): Path to save the figure.
        max_leadtime_days (int or float, optional): If set, show only the first N days of lead time (e.g. 45 for 45 days).
        show_ensemble_members (bool): If True, plot one thin faint line per ensemble member (when data has 'number' dim).
    """
    
    # 1. Setup Data & Fields: use union of variables across all experiments (and baseline)
    # so we plot only variables that exist in at least one dataset and never error on missing vars
    all_var_names = set()
    for ds in experiments.values():
        all_var_names |= {v for v in ds.data_vars if v.startswith(f"{metric}.")}
    if baseline_ds is not None:
        all_var_names |= {v for v in baseline_ds.data_vars if v.startswith(f"{metric}.")}
    if not all_var_names:
        raise ValueError(f"No variables found starting with '{metric}' in any experiment or baseline.")
    obs_vars, is_ocean = sort_variable_names_by_reference(list(all_var_names), metric)
    
    # Create clean titles for plots
    prefix_len = len(metric) + 1
    field_names = [v[prefix_len:].upper() for v in obs_vars]

    # --- Dynamic Grid Calculation ---
    n_vars = len(obs_vars)
    # Calculate ncols to make the grid as square as possible (sqrt)
    ncols = math.ceil(math.sqrt(n_vars))
    n_rows = math.ceil(n_vars / ncols)

    # 2. Setup Figure
    fig, axes = plt.subplots(n_rows, ncols, figsize=(4 * ncols, 3.5 * n_rows), constrained_layout=True)
    
    # Handle single subplot case where axes is not an array
    if n_vars == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    # Custom color palette for model experiments (persistence and climatology use fixed styling)
    MODEL_COLORS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    label_to_color_idx = {}
    color_idx = 0
    for lab in experiments:
        if lab.strip().lower() not in ('persistence', 'climatology'):
            label_to_color_idx[lab] = color_idx % len(MODEL_COLORS)
            color_idx += 1

    # Build legend entries for ALL experiments (and baseline) so legend is complete even
    # when some experiments only appear in a subset of subplots
    legend_handles = []
    if metric == "rmse":
        legend_handles.append(Line2D([], [], color='black', linewidth=2, linestyle='-', label='RMSE'))
        legend_handles.append(Line2D([], [], color='black', linewidth=2, linestyle='--', label='RMSE (ens)'))
    for label in experiments:
        if label.strip().lower() == "persistence":
            legend_handles.append(Line2D([], [], color='black', linewidth=2, linestyle=':', alpha=0.8, label=label))
        elif label.strip().lower() == "climatology":
            legend_handles.append(Line2D([], [], color='gray', linewidth=2, linestyle=':', alpha=0.8, label=label))
        else:
            color = MODEL_COLORS[label_to_color_idx[label]]
            legend_handles.append(Line2D([], [], color=color, linewidth=2, label=label))
    if baseline_ds is not None:
        bl_label = baseline_label if baseline_label else "Baseline"
        legend_handles.append(Line2D([], [], color='gray', linewidth=2, linestyle=':', alpha=0.8, label=bl_label))

    # 3. Plotting Loop (Per Variable)
    SQRT2 = np.sqrt(2)
    for idx, (field_name, var_name) in enumerate(zip(field_names, obs_vars)):
        ax = axes_flat[idx]
        sqrt2_clim_plotted = False  # add √2 × Climatology line at most once per subplot

        # Plot each experiment for this specific variable (skip if this eval doesn't have it)
        for exp_idx, (label, ds) in enumerate(experiments.items()):
            # Fallback for crps_skill: Persistence (or Climatology) may only have crps_ens; compute skill from baseline
            use_crps_ens_fallback = False
            crps_ens_fallback_var = None
            if var_name not in ds.data_vars and metric == "crps_skill" and baseline_ds is not None:
                suffix = var_name.split(".", 1)[1]
                crps_ens_var = "crps_ens." + suffix
                if (
                    label.strip().lower() in ("persistence", "climatology")
                    and crps_ens_var in ds.data_vars
                    and crps_ens_var in baseline_ds.data_vars
                ):
                    use_crps_ens_fallback = True
                    crps_ens_fallback_var = crps_ens_var
            if not use_crps_ens_fallback and var_name not in ds.data_vars:
                continue
            # Handle Lead Time (X-axis) once per dataset
            if np.issubdtype(ds.lead_time.dtype, np.timedelta64):
                x_vals = ds.lead_time.dt.days.values.copy()
            else:
                x_vals = ds.lead_time.values.copy()
            x_vals_full = x_vals.copy()
            if max_leadtime_days is not None:
                mask_lt = x_vals <= max_leadtime_days
                x_vals = x_vals[mask_lt]
            else:
                mask_lt = None

            if use_crps_ens_fallback:
                # Compute crps_skill = 1 - crps_ens / crps_ens_baseline (e.g. persistence vs climatology)
                c_ens = ds[crps_ens_fallback_var].sel(region=region)
                c_ref = baseline_ds[crps_ens_fallback_var].sel(region=region)
                for d in ['init_time', 'number']:
                    if d in c_ens.dims:
                        c_ens = c_ens.mean(dim=d, skipna=True)
                    if d in c_ref.dims:
                        c_ref = c_ref.mean(dim=d, skipna=True)
                # Align to same lead_time (use c_ens's lead_time; ref may have same)
                c_ref = c_ref.reindex_like(c_ens, method='nearest', tolerance=1e-6)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(c_ref.values != 0, c_ens.values / c_ref.values, np.nan)
                    data = xr.DataArray(1.0 - ratio, dims=c_ens.dims, coords=c_ens.coords).squeeze()
            else:
                data = ds[var_name].sel(region=region)
            # Average over init_time only (keep number for optional per-member lines)
            if 'init_time' in data.dims:
                data = data.mean(dim="init_time", skipna=True)

            # Persistence: black dotted; Climatology: gray dotted; others: custom palette
            if label.strip().lower() == "persistence":
                if 'number' in data.dims:
                    data = data.mean(dim="number", skipna=True)
                y_vals = data.values.astype(float)
                if max_leadtime_days is not None:
                    y_vals = y_vals[mask_lt]
                ax.plot(x_vals, y_vals, label=label, color='black', linewidth=2, linestyle=':', alpha=0.8)
                # When metric is rmse, also plot rmse_ens as dashed in same color
                if metric == "rmse":
                    _plot_rmse_ens_if_present(ax, ds, var_name, region, x_vals_full, mask_lt, max_leadtime_days, 'black', '--')
            elif label.strip().lower() == "climatology":
                if 'number' in data.dims:
                    data = data.mean(dim="number", skipna=True)
                y_vals = data.values.astype(float)
                if max_leadtime_days is not None:
                    y_vals = y_vals[mask_lt]
                ax.plot(x_vals, y_vals, label=label, color='gray', linewidth=2, linestyle=':', alpha=0.8)
                if not sqrt2_clim_plotted:
                    ax.plot(x_vals, y_vals * SQRT2, color='gray', linewidth=2, linestyle=':', alpha=0.6)
                    sqrt2_clim_plotted = True
                if metric == "rmse":
                    _plot_rmse_ens_if_present(ax, ds, var_name, region, x_vals_full, mask_lt, max_leadtime_days, 'gray', '--')
            else:
                # Model: optionally plot one thin faint line per ensemble member, then mean
                color = MODEL_COLORS[label_to_color_idx[label]]
                if show_ensemble_members and 'number' in data.dims:
                    # Plot each member as a slightly thin, semi-transparent line (same color)
                    for k in range(data.sizes['number']):
                        y_m = data.isel(number=k).values.astype(float)
                        if _is_geopotential_var(var_name, metric):
                            y_m = y_m / GEOPOTENTIAL_TO_METERS
                        if max_leadtime_days is not None:
                            y_m = y_m[mask_lt]
                        ax.plot(x_vals, y_m, color=color, linewidth=1.0, alpha=0.8, linestyle=':')
                    data = data.mean(dim="number", skipna=True)
                elif 'number' in data.dims:
                    data = data.mean(dim="number", skipna=True)
                y_vals = data.values.astype(float)
                if _is_geopotential_var(var_name, metric):
                    y_vals = y_vals / GEOPOTENTIAL_TO_METERS
                if max_leadtime_days is not None:
                    y_vals = y_vals[mask_lt]
                ax.plot(x_vals, y_vals, label=label, color=color, linewidth=2)
                if metric == "rmse":
                    _plot_rmse_ens_if_present(ax, ds, var_name, region, x_vals_full, mask_lt, max_leadtime_days, color, '--')
        
        # Plot baseline if provided
        if baseline_ds is not None:
            if var_name in baseline_ds.data_vars:
                # Data Extraction & Averaging for baseline
                baseline_data = baseline_ds[var_name].sel(region=region)
                # Average over init_time if present
                if 'init_time' in baseline_data.dims:
                    baseline_data = baseline_data.mean(dim="init_time", skipna=True)
                # Average over ensemble members (number) if present
                if 'number' in baseline_data.dims:
                    baseline_data = baseline_data.mean(dim="number", skipna=True)
                
                # Handle Lead Time (X-axis) for baseline
                if np.issubdtype(baseline_ds.lead_time.dtype, np.timedelta64):
                    baseline_x_vals = baseline_ds.lead_time.dt.days.values.copy()
                else:
                    baseline_x_vals = baseline_ds.lead_time.values.copy()
                baseline_x_vals_full = baseline_x_vals.copy()
                baseline_mask_lt = (baseline_x_vals <= max_leadtime_days) if max_leadtime_days is not None else None

                # Determine baseline label
                bl_label = baseline_label if baseline_label else "Baseline"
                
                # Get baseline values, keeping NaN as NaN
                baseline_y_vals = baseline_data.values.astype(float)
                # Convert geopotential to height [m]
                if _is_geopotential_var(var_name, metric):
                    baseline_y_vals = baseline_y_vals / GEOPOTENTIAL_TO_METERS

                # Optional: restrict to first N days of lead time
                if max_leadtime_days is not None:
                    mask = baseline_x_vals <= max_leadtime_days
                    baseline_x_vals = baseline_x_vals[mask]
                    baseline_y_vals = baseline_y_vals[mask]
                
                # Plot baseline as dotted line in gray (NaN values will create gaps)
                ax.plot(baseline_x_vals, baseline_y_vals, 
                       label=bl_label, color='gray', linewidth=2, 
                       linestyle=':', alpha=0.8)
                # If baseline is climatology, add √2 × climatology reference line (once per subplot)
                if not sqrt2_clim_plotted and bl_label.strip().lower() == "climatology":
                    ax.plot(baseline_x_vals, baseline_y_vals * SQRT2, color='gray', linewidth=2, linestyle=':', alpha=0.6)
                    sqrt2_clim_plotted = True
                # When metric is rmse, also plot baseline rmse_ens as dashed gray
                if metric == "rmse":
                    _plot_rmse_ens_if_present(ax, baseline_ds, var_name, region, baseline_x_vals_full, baseline_mask_lt, max_leadtime_days, 'gray', '--')

        # Add reference line at y=1 for spread_skill metric
        if metric == 'spread_skill':
            ax.axhline(y=1, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Subplot Formatting
        ax.set_title(f"{field_name}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Smart Axis Labels
        # Only bottom-most plots get X labels
        if idx >= (n_rows - 1) * ncols: 
            ax.set_xlabel("Lead time [days]")
        elif idx + ncols >= n_vars: # For incomplete last rows
            ax.set_xlabel("Lead time [days]")

        # Leftmost column gets Y labels (add [m] for geopotential variables)
        if idx % ncols == 0:
            ylabel = metric.upper().replace('_', ' ')
            if _is_geopotential_var(var_name, metric):
                ylabel += " [m]"
            ax.set_ylabel(ylabel)

        # Legend is added once after the loop using legend_handles (all experiments + baseline)

    # Add single legend to first subplot (shows all experiments, even if only some appear there)
    if legend_handles:
        axes_flat[0].legend(handles=legend_handles, loc='upper left', frameon=True, fontsize='small')

    # 4. Cleanup Empty Subplots
    for i in range(n_vars, len(axes_flat)):
        axes_flat[i].axis('off')

    # Global Title
    if metric == 'spread_skill':
        fig.suptitle(f"Spread/Skill Ratio - Region: {region.upper()}\n(>1: overdispersed, <1: underdispersed, =1: ideal)", fontsize=14)
    else:
        fig.suptitle(f"{metric.upper().replace('_', ' ')} - Region: {region.upper()}", fontsize=16)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()




def main():
    """CLI entry point for skill_curves.py"""
    parser = argparse.ArgumentParser(description="Generate skill curves from evaluation results")
    parser.add_argument("--config", "-c", help="Path to plotting config YAML file (for multi-model comparison)")
    parser.add_argument("--input", "-i", help="Path to evaluation timeseries NetCDF file (single model mode)")
    parser.add_argument("--output", "-o", required=True, help="Path to save output figure")
    parser.add_argument("--metric", "-m", default="crps_ens", help="Metric prefix to plot (default: crps_ens)")
    parser.add_argument("--region", "-r", default="global", help="Region to plot (default: global)")
    parser.add_argument("--label", "-l", default="Our Model", help="Label for the experiment (single model mode, default: Our Model)")
    parser.add_argument("--baseline", "-b", help="Path to baseline evaluation file (climatology or persistence)")
    parser.add_argument("--baseline-label", help="Label for baseline (default: auto-detect from filename)")
    parser.add_argument("--max-leadtime-days", type=float, default=None,
                        help="Show only the first N days of lead time (e.g. 45 for 45 days). Overrides config.")
    parser.add_argument("--show-ensemble-members", action="store_true",
                        help="Plot one thin faint line per ensemble member. Overrides config.")
    parser.add_argument("--no-show-ensemble-members", action="store_true",
                        help="Do not plot per-member lines (overrides config).")
    
    args = parser.parse_args()
    
    # Determine mode: config-based (multi-model) or single model
    if args.config:
        # Multi-model mode: load from config
        if args.input:
            print("WARNING: Both --config and --input provided. Using --config (multi-model mode).")
        
        experiments, _, _ = load_models_from_config(args.config)
        
        # Override metric and region from config if available
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        plotting_cfg = config.get('plotting', {})
        metric = plotting_cfg.get('metric', args.metric)
        region = plotting_cfg.get('region', args.region)
        max_leadtime_days = args.max_leadtime_days if args.max_leadtime_days is not None else plotting_cfg.get('max_leadtime_days')
        show_ensemble_members = plotting_cfg.get('show_ensemble_members', False)
        if args.show_ensemble_members:
            show_ensemble_members = True
        if args.no_show_ensemble_members:
            show_ensemble_members = False
        
        # Load baseline (command line takes precedence over config)
        baseline_ds, baseline_label = load_baseline(
            baseline_path=args.baseline,
            baseline_label=args.baseline_label,
            config=config
        )
        
    elif args.input:
        # Single model mode: load single file
        ds = xr.open_dataset(args.input)
        experiments = {args.label: ds}
        metric = args.metric
        region = args.region
        
        # Load baseline if specified
        baseline_ds, baseline_label = load_baseline(
            baseline_path=args.baseline,
            baseline_label=args.baseline_label
        )
        max_leadtime_days = args.max_leadtime_days
    else:
        parser.error("Either --config or --input must be provided")
    
    if args.config:
        show_ens = show_ensemble_members
    else:
        show_ens = args.show_ensemble_members and not args.no_show_ensemble_members
    plot_skill_curves(
        experiments,
        metric=metric,
        region=region,
        baseline_ds=baseline_ds,
        baseline_label=baseline_label,
        output_path=args.output,
        max_leadtime_days=max_leadtime_days,
        show_ensemble_members=show_ens,
    )


if __name__ == "__main__":
    main()