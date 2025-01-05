from catkit2 import TestbedProxy
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import thd
sys.path.append('C:\\Program Files\\HOLOEYE Photonics\\SLM Display SDK (Python) v3.2.2\\python')
from holoeye import slmdisplaysdk
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

''' Before Starting'''
''' On ERIS CONFIGURATION MANAGER you need to load the configuration file : ERISCalibration_NIR-153_800.00nm_2.00pi_sgl=1540.hecalib.txt'''

#BOITE NOIRE

#(s'assurer que le déterminant est positif (valeur sous la racine carrée))
def voltage(phase, alpha, beta, gamma):
    '''voltage à l'ordre 2'''
    return (-beta + np.sqrt(beta**2 - 4*gamma*(alpha-phase)))/ (2*gamma)

def voltage_deg1(phase, alpha, beta):
    '''voltage à l'ordre 1'''
    return (phase - alpha) / beta

#on crée un exemple de carte de phase : tableau numpy flottant
tableau_phase =np.zeros((1200,1920)).astype('float64')
tableau_phase[300:600,:]=1.3*np.pi
tableau_phase[900:1200,:]=np.pi

def applique_une_phase_au_slm(carte_phase, longueur_d_onde):
    """
    Cette fonction prend une carte de phase et la longueur d'onde du laser,
    génère une carte de voltage appropriée, et envoie cette carte au SLM.

    Parameters
    ----------
    carte_phase : numpy.ndarray
        Carte de phase qui doit être appliquée au SLM.
    longueur_d_onde : int
        Longueur d'onde du laser en nanomètres.

    Returns
    -------
    None
    """
    # Allumage du SLM
    slm = slmdisplaysdk.SLMInstance()
    # Vérifier si la librairie implémente la bonne version
    if not slm.requiresVersion(5):
        exit(1)
    # Ouverture du SLM et renvoie une erreur si la connexion a échoué
    error = slm.open()
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    # Vérifier la dimension de la carte de phase
    if carte_phase.shape != (1200, 1920):
        raise ValueError("La carte de phase doit être de même dimension que le SLM (1920 x 1200)")

    # Ouverture des fichiers d'étalonnage et extraction des coefficients
    fichier_633 = "./coeff_ordre2/633_800_2pi_ordre2.fits"
    fichier_705 = "./coeff_ordre2/705_800_2pi_ordre2.fits"
    fichier_785 = "./coeff_ordre2/785_800_2pi_ordre2.fits"
    
    with fits.open(fichier_633) as hdul:
        data_633 = hdul[1].data
    with fits.open(fichier_705) as hdul:
        data_705 = hdul[1].data
    with fits.open(fichier_785) as hdul:
        data_785 = hdul[1].data

    alpha_633, beta_633, gamma_633 = data_633[2][1], data_633[3][1], data_633[4][1]
    alpha_705, beta_705, gamma_705 = data_705[2][1], data_705[3][1], data_705[4][1]
    alpha_785, beta_785, gamma_785 = data_785[2][1], data_785[3][1], data_785[4][1]

    # Calcul du voltage basé sur la longueur d'onde
    if 633 <= longueur_d_onde < 705:
        V = (705 - longueur_d_onde) / (705 - 633) * voltage(carte_phase + alpha_633, alpha_633, beta_633, gamma_633) + \
            (longueur_d_onde - 633) / (705 - 633) * voltage(carte_phase + alpha_705, alpha_705, beta_705, gamma_705)
    elif 705 <= longueur_d_onde < 785:
        V = (785 - longueur_d_onde) / (785 - 705) * voltage(carte_phase + alpha_705, alpha_705, beta_705, gamma_705) + \
            (longueur_d_onde - 705) / (785 - 705) * voltage(carte_phase + alpha_785, alpha_785, beta_785, gamma_785)
    else:
        V = voltage(carte_phase + alpha_785, alpha_785, beta_785, gamma_785)

    # Affichage des cartes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im1 = axes[0].imshow(carte_phase, cmap='gray_r', origin='lower', aspect='auto', vmin=0, vmax=2*np.pi)
    axes[0].set_title("Carte de Phase pour SLM")
    im2 = axes[1].imshow(V, cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title("Carte de Voltage pour SLM")
    plt.colorbar(im1, ax=axes[0], label='Phase (radians)')
    plt.colorbar(im2, ax=axes[1], label='Voltage (%)')
    plt.tight_layout()
    plt.show()

    # Envoyer la carte de voltage au SLM
    error = slm.showData(V.astype('float64'))  # Convertit V en float64 si ce n'est pas déjà fait
    # Envoie un message d'erreur si un problème avec le SLM est rencontré
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    print("La carte de phase a bien été envoyée au SLM")


