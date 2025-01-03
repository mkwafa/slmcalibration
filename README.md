# slmcalibration

The aim of this project is to calibrate an SLM so as to apply the voltage required to reproduce the phase desired by the user.

### Data

The data used to carry out this project are in the files data_633.fits, data_705.fits and data_785.fits. They involve the acquisition of a series of 300 images 9 times for each corresponding laser. We measure the difference in intensity between a part of the image where voltage is applied and a part of the image where voltage is not applied. These intensity difference values and their associated uncertainties are grouped together in the following data files

### Fit with MCMC

We modeled the data using the MCMC method, at order 1 and order 2 using the codes:
- `mcmc_pascontactoptique_ordre1.py`
- `mcmc_pascontactoptique_ordre2.py`

The corresponding coefficients are saved in the files:
- `633_800_2.0pi_ordre1.fits`, `633_800_2.0pi_ordre2.fits`
- `705_800_2.0pi_ordre1.fits`, `705_800_2.0pi_ordre2.fits`
- `785_800_2.0pi_ordre1.fits`, `785_800_2.0pi_ordre2.fits`


