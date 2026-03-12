# DLESyM_Evals / nacc

Basic evals from our 2026 AGU Advances paper.

---

| File | Description |
|------|-------------|
| **evaluator_init.py** | Initializes `EvaluatorHPX` instances for z500, z1000, and tau300-700. Main purpose here is to remap hpx files to lat-lon (cahches lat-lon versions to <forecast-path>_varname_ll.nc). This is a necessary step for NAM, SAM and TC evals below. |
| **example_climate-var_agu-advances2026.py** | Example runner. Defines parameter dicts for seasonal cycle, blocking, NAM/SAM, and TC evaluations. Dispatches to the modules below. Configured for AGU Advances 2026 model. |
| **NAM_SAM_hpx.py** | NAM (z1000) and SAM (z500) annular modes via EOF (xeofs). Load lat-lon data, computes leading EOF, produces polar regression maps, caches results. |
| **TC_freq_hpx.py** | Tropical cyclone frequency for West Pacific. Uses z1000 and tau300-700, finds TC tracks (local z1000 minima + tau anomaly), filters short tracks, plots frequency and tracks. |
| **blocking.py** | AGP blocking index (Schiemann et al. 2020) on z500. Computes blocking frequency and std for forecast vs ERA5. Outputs polar stereographic maps; supports DLESyM and CMIP6. |
| **seasonal_cycle.py** | Seasonal cycle of z500. Remaps HEALPix to lat-lon, computes zonal mean with rolling average, plots latitude vs forecast month. |
| **xeofs_req.yaml**| Environment requirements for NAM and SAM calculation. I found the xeof dependency in `NAM_SAM_hpx.py` doesn't cooperate with my base dlesym-0.1 env.  | 
| **dlesym-0.1.yaml** | Basic requirements for evals. Does not include xeofs, which is a dependency for `NAM_SAM_hpx.py`  calculation. Everything else should work. |  
