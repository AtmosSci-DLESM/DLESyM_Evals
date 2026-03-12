try: 
    from evaluation.evaluators import EvaluatorHPX
except ImportError:
    raise ImportError("DLESyM package is a dependency of this script. Make sure it's appended to PYTHONPATH with 'export PYTHONPATH=/path/to/DLESyM'")

# z500
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    eval_variable = 'z500',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_z500_ll.nc'
)
# # z1000
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_1000.nc',
    eval_variable = 'z1000',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_z1000_ll.nc'
)
# tau300-700
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_tau300-700_new.nc',
    eval_variable = 'tau300-700',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_tau300-700_ll.nc'
)


# 1-on 1-out 12H 
# z500
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    eval_variable = 'z500',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_z500_ll.nc'
)
# z1000
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_1000.nc',
    eval_variable = 'z1000',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_z1000_ll.nc'
)
# tau300-700
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_tau300-700_new.nc',
    eval_variable = 'tau300-700',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_tau300-700_ll.nc'
)