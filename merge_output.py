# %%
from pathlib import Path
import xarray as xr

# %%
data_dir = Path("./output/250312_4750-4939_SNR266_run1_resolution")
subdirs = [
            "resolution_4.125e-4",
            "resolution_8.25e-4",
            "resolution_1.65e-3",
            "resolution_3.3e-3",
            "resolution_6.6e-3",
            "resolution_1.32e-2",
            "resolution_2.64e-2",
            "resolution_5.28e-2",
            "resolution_1.056e-1",
            "resolution_2.112e-1",
            "resolution_4.224e-1",
          ]
# subdirs = [
#             "true_column_1.65e21",
#             "true_column_3.3e21",
#             "true_column_6.6e21",
#             "true_column_13.2e21",
#         ]
concat_variable = {"resolution":
                   [4.125e-4,
                    8.25e-4,
                    1.65e-3,
                    3.3e-3,
                    6.6e-3,
                    1.32e-2,
                    2.64e-2,
                    5.28e-2,
                    1.056e-1,
                    2.112e-1,
                    4.224e-1
                    ]}
# concat_variable = {"total_column":
#                    [1.65e21,
#                     3.3e21,
#                     6.6e21,
#                     13.2e21,
#                     ]}

# %%
simspec = xr.open_mfdataset(
    [data_dir / Path(subdir) / Path("simspec.nc") for subdir in subdirs],
    concat_dim=concat_variable,
    combine="nested",
    ).assign_coords(**concat_variable)
statistics = xr.open_mfdataset(
    [data_dir / Path(subdir) / Path("statistics.nc") for subdir in subdirs],
    concat_dim=concat_variable,
    combine="nested",
    ).assign_coords(**concat_variable)
example_retr_result = xr.open_mfdataset(
    [data_dir / Path(subdir) / Path("example_retr_result.nc") for subdir in subdirs],
    concat_dim=concat_variable,
    combine="nested",
    ).assign_coords(**concat_variable)

# %%
simspec.to_netcdf(data_dir / Path("simspec.nc"))
statistics.to_netcdf(data_dir / Path("statistics.nc"))
example_retr_result.to_netcdf(data_dir / Path("example_retr_result.nc"))
# %%
