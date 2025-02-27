from pathlib import Path
import argparse
import os
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from openpathretrieval.forwardmodel import OpenPathFmShell
from openpathretrieval.backgrounds import PolynomBg
from openpathretrieval.ilsfactory import ILSFactory
from openpathretrieval.helpers import data_of_wavenumber
from openpathretrieval.retrieval_shells import RetrievalShells
from openpathretrieval.file_opener import FileOpener

def main(
    config_path: Path,
    output_dir: Path = None,
):
    """Main function to run the simulation and retrieval process.
        Parameters:
        -----------
        config_path : Path
            Path to the configuration YAML file.
        output_dir : Path, optional
            Directory to save the output files. If None, the function returns the results, by default None.
        Returns:
        --------
        tuple (optional)
            If output_dir is None, returns a tuple containing:
            - simspec : xarray.Dataset
               Dataset containing the noise-free spectrum and spectrum with noise.
            - results : xarray.Dataset
                Dataset containing the retrieval results for each run.
            - retr_result : xarray.Dataset
                Dataset containing the retrieval result for the last run.
        """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    wn_start = config["retrwindows"]["window1"]["nu_start"]
    wn_stop = config["retrwindows"]["window1"]["nu_stop"]
    wn_step = config["retrwindows"]["window1"]["nu_step"]

    xsdb_path = Path(config["retrwindows"]["window1"]["xsdb_path"])

    ils_model = config["retrwindows"]["window1"]["ils"]["type"]
    resolution = config["retrwindows"]["window1"]["ils"]["resolution"]

    temperature = config["atmosphere"]["temperature"]
    pressure = config["atmosphere"]["pressure"]

    snr = config["statistics"]["snr"]
    number_of_runs = config["statistics"]["number_of_runs"]

    true_columns = config["true_column"]

    # Generate additional input variables
    simulation_grid = np.arange(wn_start-50, wn_stop+50, wn_step)
    
    # Generate forward modeled spectrum

    noise_free_spectrum = model_noise_free_spectrum(
        simulation_grid,
        xsdb_path,
        ils_model,
        true_columns, 
        temperature,
        pressure,
        resolution=resolution,
    )

    gas_results = {key: [] for key in config["absorber"].keys()}
    gas_results_err = {f"{key}_err": [] for key in config["absorber"].keys()}

    # Generate multiple spectra with noise and retrieve them
    for n in tqdm(range(number_of_runs), desc="Retrieval progress", unit="run"):
        spectrum = add_noise_to_spectrum(
            noise_free_spectrum,
            snr,
            add_noise_window=True)
        retr_result = retrieve_spectrum(
            config,
            spectrum,
            temperature=temperature,
            pressure=pressure,
            )
        for key in gas_results.keys():
            gas_results[key].append(retr_result[key].values)
        for key in gas_results_err.keys():
            gas_results_err[key].append(retr_result[key].values)

    data_vars_simspec = {
        "noise_free_spectrum": noise_free_spectrum,
        "spectrum_with_noise": spectrum.sel(
            wavenumber=slice(simulation_grid.min(), simulation_grid.max())
        ),
    }
    coords_simspec = {"wavenumber": simulation_grid}
    simspec = xr.Dataset(data_vars=data_vars_simspec, coords=coords_simspec)

    run_number = range(number_of_runs)
    data_vars_result = {
        **{key: xr.DataArray(
            gas_results[key],
            {"run_number": run_number}
            )
            for key in gas_results.keys()
        },
        **{key: xr.DataArray(
            gas_results_err[key],
            {"run_number": run_number}
            )
            for key in gas_results_err.keys()
        },
    }
    coords_result = {"run_number": run_number}
    results = xr.Dataset(data_vars=data_vars_result, coords=coords_result)

    if output_dir is not None:
        simspec_file_dir = output_dir / Path("simspec.nc")
        statistics_output_dir = output_dir / Path("statistics.nc")
        retr_result_dir = output_dir / Path("example_retr_result.nc")

        simspec.to_netcdf(simspec_file_dir)
        results.to_netcdf(statistics_output_dir)
        retr_result.to_netcdf(retr_result_dir)
    else:
        return simspec, results, retr_result
    

def model_noise_free_spectrum(
    wn_grid: np.ndarray,
    xsdb_path: Path,
    ils_model: str,
    total_columns: dict,
    temperature: float,
    pressure: float,
    resolution: float = 0.6,
) -> xr.DataArray:
    """Modeling a noise free open path spectrum.

        Parameters
        ----------
        wn_grid : np.ndarray
            Wavenumber grid for which the spectrum should be returned.
        xsdb_path : Path
            Path to the cross section database.
        ils_model : str
            Instrument line shape model. Either 'NBM' or 'Delta'.
        total_columns : dict
            Total columns of the absorbing species.
        temperature : float	
            Temperature of the atmosphere.
        pressure : float
            Pressure of the atmosphere.
        resolution : float, optional
            Resolution of the instrument line shape as set in OPUS
            (0.9 * opd_max). Only relevant for ILS model 'NBM', by default 0.6.

        Returns
        -------
        xr.DataArray
            A forward simulated open path spectrum.

        """
    wavenumber_array = data_of_wavenumber(wn_grid, wn_grid)
    wn_step = wn_grid[1] - wn_grid[0]

    if ils_model == "NBM":
        ils = ILSFactory.norton_beer_medium(
            wavenumber_step=wn_step,
            width=10,
            resolution=resolution,
        )	
    elif ils_model == "delta":
        ils = ILSFactory.delta_peak()
    else:
        raise ValueError("ILS model must be 'NBM' or 'Delta'")  
     
    xsdb = xr.open_dataset(xsdb_path, engine="h5netcdf")
    bg_type = PolynomBg

    fm = OpenPathFmShell(
        wavenumber_array=wavenumber_array,
        ils=ils,
        xsdb=xsdb,
        bg_type=bg_type,
    )

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


def add_noise_to_spectrum(
    spectrum : xr.DataArray,
    snr : float,
    add_noise_window : bool = False,
) -> xr.DataArray:
    """Add gaussian noise to a spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            Simulated noise free spectrum.
        snr : float
            Signal to noise ratio.
        add_noise_window : bool, optional
            Add a noise window from 3000cm^-1 to 4000cm^-1 which the retrieval
            might require to work, by default False.
    
        Returns
        -------
        xr.DataArray
            A spectrum with added gaussian noise.

        """
    noise = np.random.normal(0, 1/snr, size=len(spectrum))
    spectrum = spectrum + noise
    if add_noise_window:
        wn_spacing = spectrum.wavenumber[1] - spectrum.wavenumber[0]
        noise_window = np.arange(3000, 4000, wn_spacing)
        noise = np.random.normal(0, 1/snr, size=len(noise_window))
        noise_window_array = xr.DataArray(data=noise, coords={"wavenumber": noise_window})
        spectrum = xr.concat([spectrum, noise_window_array], "wavenumber")
    return spectrum


def down_sample_spectrum(
    spectrum : xr.DataArray,
    wn_spacing : float,
) -> xr.DataArray:
    """Downsample a spectrum to a given wavenumber spacing.

        Parameters
        ----------
        spectrum : xr.DataArray
            A spectrum to be downsampled.
        wn_spacing : float
            Wavenumber spacing of the downsampled spectrum.
    
        Returns
        -------
        xr.DataArray
            A downsampled spectrum.

        """
    nu_min = spectrum.wavenumber.min().values
    nu_max = spectrum.wavenumber.max().values
    new_grid = np.arange(nu_min, nu_max, wn_spacing)
    spectrum_downsampled = spectrum.interp(wavenumber=new_grid)
    return spectrum_downsampled


def retrieve_spectrum(
    config : dict,
    simulated_measurement : xr.DataArray,
    temperature : float = 300,
    pressure : float = 1015,
) -> xr.Dataset:
    """Run retrieval on a simulated measurement.

        Parameters
        ----------
        config : dict
            Config dictionary.
        simulated_measurement : xr.DataArray
            Simulated measurement spectrum.
        temperature : float, optional
            Temperature of the atmosphere, by default 300.
        pressure : float, optional
            Pressure of the atmosphere, by default 1015.
    
        Returns
        -------
        xr.Dataset
            A dataset containing the retrieval result.
        """
    #config = FileOpener.read_config(config_path)
    retr_result = RetrievalShells.retrieve_measurement(
        measured_spectrum=simulated_measurement,
        retr_config=config,
        environment_data={"temperature": temperature, "pressure": pressure},
        meta_data={"time": np.datetime64("1900")},
    )
    return retr_result


def print_statistics(data_array : xr.DataArray):
    """Print statistics of a data array.

        Parameters
        ----------
        data_array : xr.DataArray
            A data array for which the statistics should be printed.

        """
    print(f"{data_array.mean().values} +- {data_array.std().values}")
    print(f"{data_array.std().values / data_array.mean().values * 100:.2} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to simulate and retrieve open path spectra.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output.")

    args = parser.parse_args()

    # Optional: Check if output directory exists, create if not
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Call main function
    main(args.config_file, args.output_dir)
