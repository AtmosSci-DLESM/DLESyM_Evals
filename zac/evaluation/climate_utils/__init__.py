"""
climate_utils

Climate analysis utilities for evaluating 50-year climate forecasts.

This package contains individual analysis functions that can be called by
evaluate_climate.py. Each analysis function should:
  - Accept (inference_file: xr.Dataset, target_file: xr.Dataset, **kwargs) as signature
  - Handle errors gracefully and log appropriately
  - Return Optional[xr.DataArray] or None
  - Compare model predictions (inference_file) against observations (target_file)
"""

