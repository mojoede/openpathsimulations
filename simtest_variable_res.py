# %%
from pathlib import Path

import numpy as np
import plotly.express as px
import xarray as xr

from openpathretrieval.forwardmodel import OpenPathFmShell
from openpathretrieval.backgrounds import PolynomBg
from openpathretrieval.ilsfactory import ILSFactory
from openpathretrieval.helpers import data_of_wavenumber
from openpathretrieval.retrieval_shells import RetrievalShells
from openpathretrieval.file_opener import FileOpener


# %%
def noise_free_spectrum(resolution=0.6):
    # wavenumbers = np.arange(4710, 5150, 0.005)
    wavenumbers = np.arange(6100, 6450, 0.005)
    wavenumber_array = data_of_wavenumber(wavenumbers, wavenumbers)
    ils = ILSFactory.norton_beer_medium(
        wavenumber_step=0.005,
        width=10,
        resolution=resolution,
    )
    xsdb = xr.open_dataset(Path(
        r"C:\Daten\Hitran\forwardmodel\xsections\combined\\"
        r"5800-6600-CO2_626_628_636-CH4_211_311-H2O_161_162_181.hdf5"
        # r"4700-5300-CO2_all-H2O_all-HDO.hdf5"
    ))
    bg_type = PolynomBg
    fm = OpenPathFmShell(
        wavenumber_array=wavenumber_array,
        ils=ils,
        xsdb=xsdb,
        bg_type=bg_type,
    )
    total_columns = {
        "CO2": 3.3e21,
        "H2O": 7.5e22,
    }
    temperature = 300.
    pressure = 1015.
    background_coeff = [1, 0, 0]
    stretch_coeff = [0, 1]
    simspec = fm.spectrum(
        total_columns=total_columns,
        temperature=temperature,
        pressure=pressure,
        background_coeff=background_coeff,
        stretch_coeff=stretch_coeff,
    )
    return simspec.dropna("wavenumber")


def add_noise_to_spectrum(spectrum, snr, add_noise_window=False):
    noise = np.random.normal(0, 1/snr, size=len(spectrum))
    spectrum = spectrum + noise
    if add_noise_window:
        noise_window = np.arange(3000, 4000, 0.005)
        noise = np.random.normal(0, 1/snr, size=len(noise_window))
        noise_window_array = xr.DataArray(data=noise, coords={"wavenumber": noise_window})
        spectrum = xr.concat([spectrum, noise_window_array], "wavenumber")
    return spectrum


def down_sample_spectrum(spectrum, spacing):
    nu_min = spectrum.wavenumber.min().values
    nu_max = spectrum.wavenumber.max().values
    new_grid = np.arange(nu_min, nu_max, spacing)
    spectrum_downsampled = spectrum.interp(wavenumber=new_grid)
    return spectrum_downsampled


def retrieve_spectrum(simulated_measurement):
    result = RetrievalShells.retrieve_measurement(
        measured_spectrum=simulated_measurement,
        retr_config=FileOpener.read_config(Path("sim-test2.yaml")),
        environment_data={"temperature": 300, "pressure": 1015},
        meta_data={"time": np.datetime64("1900")},
    )
    return result


def collect_statistic(number_of_runs, snr, resolution=0.6):
    run_number = []
    CO2_results = []
    H2O_results = []
    for n in range(number_of_runs):
        spectrum = noise_free_spectrum(resolution=resolution)
        spectrum = down_sample_spectrum(spectrum, resolution/2)
        spectrum = add_noise_to_spectrum(spectrum, snr, add_noise_window=True)
        result = retrieve_spectrum(spectrum)
        run_number.append(n)
        CO2_results.append(result["CO2_1"].values)
        H2O_results.append(result["H2O_1"].values)
    data_vars = {
        "CO2": xr.DataArray(CO2_results, {"run_number": run_number}),
        "H2O": xr.DataArray(H2O_results, {"run_number": run_number}),
    }
    coords = {"run_number": run_number}
    results = xr.Dataset(data_vars=data_vars, coords=coords)
    return results


# %%
def print_statistics(da):
    print(f"{da.mean().values} +- {da.std().values}")
    print(f"{da.std().values / da.mean().values * 100:.2} %")


# %%
snr200 = xr.open_dataset(Path("2u0_snr0200.hdf5"))
snr800 = xr.open_dataset(Path("2u0_snr0800.hdf5"))
snr200_downsampled = xr.open_dataset(Path("2u0_snr0200_downsampled.hdf5"))
snr800_downsampled = xr.open_dataset(Path("2u0_snr0800_downsampled.hdf5"))
snr800_1u6 = xr.open_dataset(Path("1u6_snr0800_downsampled_2.hdf5"))
# %%
px.histogram(x=snr200.CO2)
# %%
px.histogram(x=snr800.CO2)
# %%
px.histogram(x=snr200.H2O)
# %%
px.histogram(x=snr800.H2O)
# %%
px.histogram(x=snr200_downsampled.CO2)
# %%
px.histogram(x=snr800_downsampled.CO2)
# %%
px.histogram(x=snr200_downsampled.H2O)
# %%
px.histogram(x=snr800_downsampled.H2O)
# %%
# snr200_downsampled = collect_statistic(100, 200)
# %%
# snr200_downsampled.to_netcdf(Path("2u0_snr0200_downsampled.hdf5"))
# %%
print_statistics(snr200.CO2)
print_statistics(snr800.CO2)
print_statistics(snr200.H2O)
print_statistics(snr800.H2O)
# %%
print_statistics(snr200_downsampled.CO2)
print_statistics(snr800_downsampled.CO2)
print_statistics(snr800_1u6.CO2)
print_statistics(snr200_downsampled.H2O)
print_statistics(snr800_downsampled.H2O)
print_statistics(snr800_1u6.H2O)
# %%
snr800_1u6_06 = collect_statistic(1000, 800, 0.6)
# %%
snr800_1u6_06.to_netcdf(Path("1u6_snr0800_res06.hdf5"))
# %%
snr800_1u6_03 = collect_statistic(1000, 800, 0.3)
# %%
snr800_1u6_03.to_netcdf(Path("1u6_snr0800_res03.hdf5"))
# %%
snr400_1u6_03 = collect_statistic(1000, 400, 0.3)
# %%
snr400_1u6_03.to_netcdf(Path("1u6_snr0400_res03.hdf5"))
# %%
snr1200_1u6_09 = collect_statistic(1000, 1200, 0.9)
# %%
snr1200_1u6_09.to_netcdf(Path("1u6_snr1200_res09.hdf5"))
# %%
print_statistics(snr800_1u6_06.CO2)
# %%
print_statistics(snr800_1u6_03.CO2)
# %%
print_statistics(snr400_1u6_03.CO2)
# %%
print_statistics(snr1200_1u6_09.CO2)

# %%
for i in range(10):
    print_statistics(snr800_1u6_06.CO2[i*100:(i+1)*100])
# %%
snr800_1u6_06 = xr.open_dataset(Path("1u6_snr0800_res06.hdf5"))
snr1200_1u6_09 = xr.open_dataset(Path("1u6_snr1200_res09.hdf5"))
# %%
