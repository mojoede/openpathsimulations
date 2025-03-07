# %%
from scipy.optimize import curve_fit
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

%matplotlib widget
from openpathsimulation import read_background_spectrum

background_dir = Path("/home/msindram/Data_PhD/DCS/20250304/Polynomial_fit.mat")
background = read_background_spectrum(background_dir, ".mat", "polynomial_fit")
normalized_wavenumber = background.wavenumber.values/background.wavenumber.mean().item() - 1
print(normalized_wavenumber)

def polynomial(x, *coefficients):
    y = 0
    for i in range(len(coefficients)):
        y += coefficients[i] * x ** i
    return y

popt, pcov = curve_fit(polynomial, normalized_wavenumber, background.data, p0=[  7.59904242e-01,  5.03287741e+00,  9.38525822e+02,  6.66967214e+04,
       -5.36832861e+06, -1.85781802e+08,  1.25857901e+10,  1.19415060e+11,
       -1.30232318e+13,  4.90446426e+13])

fig, ax = plt.subplots()
ax.plot(background.wavenumber, background.data, label="Background spectrum")
ax.plot(background.wavenumber, polynomial(normalized_wavenumber, *popt), label="Polynomial fit")
ax.legend()
# %%
