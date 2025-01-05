# SLM Calibration

The aim of this project is to calibrate an SLM so as to apply the voltage required to reproduce the phase desired by the user.

### Data

The data used to carry out this project are stored in the folder `data`, they are the files :
- `data_633.fits`
- `data_705.fits`
- `data_785.fits`

These files contain a series of 300 images captured 9 times for each corresponding laser. We measure the difference in intensity between a part of the image where voltage is applied and a part of the image where voltage is not applied. The intensity difference values and their associated uncertainties are grouped together in these data files. The first line si the voltage rate, the second lline the uncertainty of the intensity difference measurement and the last line is the intensity difference.
These data were recorded as part of an experiment. We set up a Michelson where the SLM is in place of a plane mirror, we position ourselves at the flat tint of the michelson and we apply a voltage rate ranging from 0 to 1.

### Fit with MCMC

We modeled the data using the MCMC method, at order 1 and order 2 using the codes:
- `mcmc_pascontactoptique_ordre1.py`
- `mcmc_pascontactoptique_ordre2.py`

The corresponding coefficients are saved in the folder `coeff_ordre1` and `coeff_ordre2`, they are the files :
- `633_800_2pi_ordre1.fits`, `633_800_2pi_ordre2.fits`
- `705_800_2pi_ordre1.fits`, `705_800_2pi_ordre2.fits`
- `785_800_2pi_ordre1.fits`, `785_800_2pi_ordre2.fits`

### Black Box

The code `boite_noire_final.py` is responsible for determining the voltage to be applied to the SLM based on a provided phase table. This conversion process utilizes coefficients determined through the MCMC method. For improved precision, it is recommended to use the coefficient files ending in `*_ordre2`.

To produce the requested phase on the SLM, use the `applique_une_phase_au_slm` function contained within `boite_noire_final.py`.

#### Function `applique_une_phase_au_slm`

This function takes a phase map and a laser wavelength, generates an appropriate voltage map, and sends this map to the SLM.

**Parameters:**
- `phase_map` (`numpy.ndarray`): The phase map to be applied to the SLM.
- `wavelength_length` (`int`): The laser wavelength in nanometers.

**Returns:**
- `None`: The function does not return any value but sends the voltage map to the SLM for the phase modulation.





