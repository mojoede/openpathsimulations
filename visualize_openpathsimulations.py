# %% Imports
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from openpathsimulation import read_background_spectrum

%matplotlib widget
# %% Enter and load data
true_columns = {
    "CO2_626_1": 4.684214813874019e+21,
    "CO2_636_1": 4.1307671963965755e+21,
    "CO2_628_1": 4.99823947679575e+21,
    "H2O_1": 6.908656067479201e+20,
}

R_CO2_636_VPDB = 0.0111802
hitran_12C_abundance = 0.984204
hitran_13C_abundance = 0.011057

data_dir = Path("./output/250307_4750-4939_SNR266_run1_skewed-noise")
simspec = xr.load_dataset(data_dir / Path("simspec.nc"))
results = xr.load_dataset(data_dir / Path("statistics.nc"))
retr_result = xr.load_dataset(data_dir / Path("example_retr_result.nc"))

# %% Calculate relative errors
rel_columns = {}
retr_err = {}
for key in true_columns.keys():
    rel_columns[key] = results[key].values / true_columns[key]
    retr_err[key] = results[f"{key}_err"].values

# %% Calculate delta values
def calc_delta(
        R_standard: float,
        column_abund_sample: np.ndarray,
        column_abund_sample_err: np.ndarray,
        hitran_ratio_abund_sample: float,
        column_rare_sample: np.ndarray,
        column_rare_sample_err: np.ndarray,
        hitran_ratio_rare_sample: float,
        ) -> tuple[np.ndarray, np.ndarray]:
    column_abund_sample = column_abund_sample * hitran_ratio_abund_sample
    column_abund_sample_err = column_abund_sample_err * hitran_ratio_abund_sample
    column_rare_sample = column_rare_sample * hitran_ratio_rare_sample
    column_rare_sample_err = column_rare_sample_err * hitran_ratio_rare_sample
    R_sample = column_rare_sample / column_abund_sample

    delta = (R_sample / R_standard - 1) * 1000
    delta_err = column_rare_sample / (column_abund_sample * R_standard) * 1000 * np.sqrt((column_rare_sample_err / column_rare_sample)**2 + (column_abund_sample_err / column_abund_sample)**2)
    return delta, delta_err

delta_13C, delta_13C_err = calc_delta(
    R_CO2_636_VPDB,
    results["CO2_626_1"].values,
    results["CO2_626_1_err"].values,
    hitran_12C_abundance,
    results["CO2_636_1"].values,
    results["CO2_636_1_err"].values,
    hitran_13C_abundance,
)

# %% Load measured reference spectrum
refspec_dir = Path("/home/msindram/Data_PhD/DCS/20250304/Spectral_trans.mat")
refspec = read_background_spectrum(refspec_dir, ".mat", "Spectral_trans")

# %% Plot Spectrum fit and residue
fig1, ax1 = plt.subplots()
simspec.spectrum_with_noise.plot(ax=ax1, label="Spectrum + noise")
refspec_norm = refspec / 0.72
refspec_norm.plot(ax=ax1, label="Measured spectrum")
simspec.noise_free_spectrum.plot(ax=ax1, label="Forward modeled spectrum")
ax1.set_xlabel("wavenumber $\\left[\\mathrm{cm}^{-1}\\right]$")
ax1.set_ylabel("transmission")
ax1.set_title("Simulated spectrum")
ax1.legend()
ax1.grid()

fig2, ax2 = plt.subplots()
simspec.spectrum_with_noise.plot(ax=ax2, label="input spectrum")
retr_result.fitted_spectrum.plot(ax=ax2, label="fitted spectrum")
ax2.set_xlabel("wavenumber $\\left[\\mathrm{cm}^{-1}\\right]$")
ax2.set_ylabel("transmission")
ax2.set_title("Fitted spectrum")
ax2.legend()
ax2.grid()

noise = simspec.noise_free_spectrum - simspec.spectrum_with_noise
residual = retr_result.fitted_spectrum - retr_result.measured_spectrum
fig3, ax3 = plt.subplots()
noise.plot(ax=ax3, label="noise")
residual.plot(ax=ax3, label="residual")
ax3.set_xlabel("wavenumber $\\left[\\mathrm{cm}^{-1}\\right]$")
ax3.set_ylabel("transmission")
ax3.set_title("Noise and retrieval residual")
ax3.legend()
ax3.grid()
plt.show()

# %% Plot retrieval result histograms
keys = list(rel_columns.keys())
for i, key in enumerate(keys):
    bias = rel_columns[keys[i]].mean()
    scatter = rel_columns[keys[i]].std()
    mean_rel_retr_err = retr_err[keys[i]].mean() / true_columns[keys[i]]
    fig4, ax4 = plt.subplots()
    ax4.hist(
        rel_columns[keys[i]],
        color = "b",
        label = f"""mean: {bias:.5f}
        std: {scatter:.5f}
        mean relative retrieval error: {mean_rel_retr_err:.5f}""",
    )
    ax4.set_xlabel("relative deviation from true column")
    ax4.set_ylabel("counts")
    ax4.set_title(keys[i][:-2])
    legend = ax4.legend(loc="upper right", handletextpad=2.0)
    for text in legend.get_texts():
        text.set_ha("right")  # Align text to the right
    ax4.grid()

# %% Plot delta 13C result histogram
mean = delta_13C.mean()
true_delta = ((true_columns["CO2_636_1"] * hitran_13C_abundance / (true_columns["CO2_626_1"] * hitran_12C_abundance)) / R_CO2_636_VPDB - 1) * 1000
scatter = delta_13C.std()
propagated_error = delta_13C_err.mean()
fig5, ax5 = plt.subplots()
ax5.hist(delta_13C,
         color="b",
         label = f"""true $\\delta^{{13}}C$: {true_delta:.2f}‰
         mean $\\delta^{{13}}C$: {mean:.2f}‰
         std: {scatter:.2f}‰
         propagated error: {propagated_error:.2f}‰"""
    )
ax5.set_xlabel("$\\delta^{13}C$ [‰]")
ax5.set_ylabel("counts")
ax5.set_title("$\\delta^{13}C$")
legend = ax5.legend(loc="upper right", handletextpad=2.0)
for text in legend.get_texts():
    text.set_ha("right")  # Align text to the right
ax5.grid()

# %%
