import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from astropy.io import fits

# %matplotlib inline
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 16

# DATA:
#output_fits_1 = "./data/data633_800.fits"
#output_fits_1 = "./data/data705_800.fits"
output_fits_1 = "./data/data785_800.fits"

# Charger le fichier FITS
with fits.open(output_fits_1) as hdul:
    data = hdul[1].data
    print(data)

    i = np.array([ligne[0] for ligne in data])
    yerr = np.array([ligne[1] for ligne in data])
    di = np.array([ligne[2] for ligne in data])

# Vérification des données
print("Vérification des données d'entrée:")
print("i contient NaN :", np.isnan(i).any())
print("di contient NaN :", np.isnan(di).any())
print("yerr contient NaN :", np.isnan(yerr).any())
print("i contient inf :", np.isinf(i).any())
print("di contient inf :", np.isinf(di).any())
print("yerr contient inf :", np.isinf(yerr).any())

# Nettoyage des données (si nécessaire)
mask = np.isfinite(i) & np.isfinite(di) & np.isfinite(yerr)
i = i[mask]
di = di[mask]
yerr = yerr[mask]


# PARAMETRE MCMC

#nouvelle suggestion de modèle :
def model(theta, i):
    A, phi1, alpha, beta = theta
    model = A * ( np.cos(phi1) - np.cos( alpha + beta * i - phi1))
    return model

def lnlike(theta, x, y, yerr):
    A, phi1, alpha, beta = theta
    LnLike = -1 / 2 * np.sum(((y - model(theta, x)) / yerr) ** 2)
    return LnLike


def lnprior(theta):
    A, phi1, alpha , beta = theta
    if 20 < A < 25 and 0 <= phi1 <= 1 and -1 < alpha < 0  and  6 < beta < 8:
        return 0.0
    else:
        return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not lp == 0.0:
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)  # recall if lp not -inf, its 0, so this just returns likelihood


# RUN MCMC
data = (i, di, yerr)

# set nwalkers
nwalkers = 100
# set niter
niter = 6000
initial = np.array([22,  0.25 , -0.58, 6.5])  # initial for A, phi1, alpha, beta and gamma
ndim = len(initial)
p0 = [np.array(initial) + 0.1 * np.random.randn(ndim) for i in range(nwalkers)]


def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state


sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)


def plotter(sampler, i=i,di=di):
    plt.ion()
    plt.errorbar(i, di, yerr = yerr, label='Diff intensites en fonction de i')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(i, model(theta, i), color="r", alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('Image i')
    plt.ylabel(r'$\Delta$ Diff intensites')
    plt.grid()
    plt.legend()
    plt.show()


# sampler= main(p0)
plotter(sampler)

samples = sampler.flatchain
samples[np.argmax(sampler.flatlnprobability)]


theta_max  = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max,i)

#extraction des incertitudes à 1 sigma:
A_samples = np.array(samples)[:,0]
phi1_samples = np.array(samples)[:,1]
alpha_samples = np.array(samples)[:,2]
beta_samples = np.array(samples)[:,3]

#delta_samples = np.array(samples)[:,5]

#quantiles:
#index = 0 pour A, 1 pour alpha etc

def calcul_erreurs(samples, index):
    Q1 = np.quantile(samples, 0.16)
    Q2 = np.quantile(samples, 0.84)
    Q = np.quantile(samples, 0.5)
    erreur_inf = Q - Q1
    erreur_sup = Q2 - Q
    return erreur_inf, erreur_sup


# Récupérer les erreurs pour chaque paramètre
A_err_inf, A_err_sup = calcul_erreurs(A_samples, 0)
phi1_err_inf, phi1_err_sup = calcul_erreurs(phi1_samples,1)
alpha_err_inf, alpha_err_sup = calcul_erreurs(alpha_samples, 2)
beta_err_inf, beta_err_sup = calcul_erreurs(beta_samples, 3)

#delta_err_inf, delta_err_sup = calcul_erreurs(delta_samples, 5)


#PLOT FINAL
plt.figure()
plt.plot(i, di, label='Diff intensites en fonction de i', color = 'blue')
plt.fill_between(i, di+yerr, di-yerr, color = 'blue', alpha = 0.1)
plt.plot(i,best_fit_model,label='Fit Likelihood \nA = '+str(np.round(theta_max[0], 2))+r", $\phi_1$ ="+str(np.round(theta_max[1], 2))
                                +r", $\alpha$ = "+str(np.round(theta_max[2], 2))+r", $\beta$ = "+str(np.round(theta_max[3], 2)),
                                color = 'salmon')
plt.xlabel(r'Fraction de Voltage')
plt.ylabel(r'Différence intensité $\Delta I$')
plt.title("'Pas contact optique'")
plt.grid()
plt.legend(loc='upper right')
plt.savefig('./data/785_ordre1.png')
plt.show()
print('Theta max: ',theta_max)




# Calcul des erreurs (à partir des quantiles)
def calcul_erreurs(samples, index):
    Q1 = np.quantile(samples, 0.16)
    Q2 = np.quantile(samples, 0.84)
    Q = np.quantile(samples, 0.5)
    erreur_inf = Q - Q1
    erreur_sup = Q2 - Q
    return erreur_inf, erreur_sup


# Récupérer les erreurs pour chaque paramètre
A_err_inf, A_err_sup = calcul_erreurs(A_samples, 0)
phi1_err_inf, phi1_err_sup = calcul_erreurs(phi1_samples,1)
alpha_err_inf, alpha_err_sup = calcul_erreurs(alpha_samples, 2)
beta_err_inf, beta_err_sup = calcul_erreurs(beta_samples, 3)
#gamma_err_inf, gamma_err_sup = calcul_erreurs(gamma_samples, 4)
#delta_err_inf, delta_err_sup = calcul_erreurs(delta_samples, 5)

# Les valeurs des coefficients et leurs erreurs
coefficients = {
    "A": {"value": theta_max[0], "err_inf": A_err_inf, "err_sup": A_err_sup},
    "phi1": {"value": theta_max[1], "err_inf":phi1_err_inf, "err_sup": phi1_err_sup},
    "alpha": {"value": theta_max[2], "err_inf": alpha_err_inf, "err_sup": alpha_err_sup},
    "beta": {"value": theta_max[3], "err_inf": beta_err_inf, "err_sup": beta_err_sup},
    #"gamma": {"value": theta_max[4], "err_inf": gamma_err_inf, "err_sup": gamma_err_sup},
    #"delta": {"value": theta_max[5], "err_inf": delta_err_inf, "err_sup": delta_err_sup},
}

# Créer des colonnes FITS
columns = [
    fits.Column(name='Parametre', format='20A', array=['A', 'phi1', 'alpha', 'beta']),
    fits.Column(name='Valeur', format='E', array=[coefficients[param]["value"] for param in coefficients]),
    fits.Column(name='Erreur inferieure', format='E', array=[coefficients[param]["err_inf"] for param in coefficients]),
    fits.Column(name='Erreur superieure', format='E', array=[coefficients[param]["err_sup"] for param in coefficients]),
]

# Créer une table binaire FITS
hdu = fits.BinTableHDU.from_columns(columns)

# Ajouter un commentaire ou un header
hdu.header['FICHIER'] = "785nm 800nm 2.0"
hdu.header['COMMENT'] = "Coefficients et leurs erreurs pour les parametres"

# Sauvegarder dans un fichier FITS
output_fits ='./coeff_ordre1/785_800_2pi_ordre1.fits'
hdu.writeto(output_fits, overwrite=True)

print(f"Les coefficients et leurs erreurs ont été écrits dans le fichier FITS : {output_fits}")


#corner : paramètres avec quantiles
labels = ['A',r'$\phi_1$',r'$\alpha$',r'$\beta$',r'$\gamma$']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
plt.savefig('./data/785_ordre1_corner.png')
plt.show()
