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


## Allumage et acquisition d'images
slm = slmdisplaysdk.SLMInstance()

## Check if the library implements the required version
if not slm.requiresVersion(5):
    exit(1)

#ouvre le SLM et renvoie une erreur si la connexion a échouée
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

#connexion à la camera (si besoin)
print('start')
testbed = TestbedProxy('127.0.0.1', 8768)               # Establish connection to the testbed server.

cam = testbed.corono_camera                           # Start up / connect to the camera.
cam.exposure_time = 109974.681  #pour 633nm

#on prend une photo avant d'appliquer un voltage:
data_photo=np.zeros((1200,1920),dtype=float)
error = slm.showData(data_photo)
time.sleep(0.1)
array = list(cam.take_raw_exposures(num_exposures=1))[0]
fits.writeto('D:\\slm_osae_2024\\resultats\\test_boite_noire\\image_sansvoltage.fits',array,overwrite=True)

print('end image')


#(s'assurer que le déterminant est positif (valeur sous la racine carrée))
def voltage(phase, alpha, beta, gamma):
    '''voltage à l'ordre 2'''
    return (-beta + np.sqrt(beta**2 - 4*gamma*(alpha-phase)))/ (2*gamma)

def voltage_deg1(phase, alpha, beta):
    '''voltage à l'ordre 1'''
    return (phase - alpha) / beta


#BOITE NOIRE

#on crée un exemple de carte de phase : tableau numpy flottant
tableau_phase =np.zeros((1200,1920)).astype('float64')
tableau_phase[300:600,:]=1.3*np.pi
tableau_phase[900:1200,:]=np.pi

#Fonction qui prendra un fichier contenant la carte de phase, longueur d'onde du laser et renvoie un tableau du taux de voltage à appliquer
def carte_de_voltage(carte_phase, longueur_d_onde):    

    dimension_carte = carte_phase.shape
    
    #dimension SLM [1920 x 1200]
    ligne_SLM = 1200
    colonne_SLM = 1920

    if dimension_carte[0] != ligne_SLM or dimension_carte[1] != colonne_SLM :
        raise ValueError("La carte de phase doit être de même dimension que le SLM (1920 x 1200)")

    #moduler toutes les valeurs de phase à 2pi si les valeurs sont supérieures à 2pi
    #carte_phase = np.mod(carte_phase, 2*np.pi)

    #on ne choisit qu'un seul étalonnage pour toutes les longueurs d'onde : 800nm 2pi
    #ouverture et lecture du fichier d'étalonnage correspondant au longueur_d_onde
    #changer l'adresse si les fichiers on changé de dossier
    fichier_633 = "D:/slm_osae_2024/resultats/nouveau test 633/633_800_2.0pi_ordre1.fits"
    fichier_705 = "D:/slm_osae_2024/resultats/nouveau test 705/705_800_2.0pi_ordre1.fits"
    fichier_785 = "D:/slm_osae_2024/resultats/nouveau test 785/785_800_2.0pi_ordre1.fits"
    

    #ouverture de chaque fichier d'étalonnage 
    with fits.open(fichier_633) as hdul:
        data_633 = hdul[1].data
    with fits.open(fichier_705) as hdul:
        data_705 = hdul[1].data
    with fits.open(fichier_785) as hdul:
        data_785 = hdul[1].data

    # extraction des coefficients alpha, beta et gamma
    alpha_633, beta_633 = data_633[2][1], data_633[3][1]
    alpha_705, beta_705 = data_705[2][1], data_705[3][1]
    alpha_785, beta_785 = data_785[2][1], data_785[3][1]
    
    #conversion phase --> voltage 
    #ici à l'ordre 1
    if 633 <= longueur_d_onde < 705:
        V = (705 - longueur_d_onde) / (705 - 633) * voltage_deg1(tableau_phase + alpha_633, alpha_633, beta_633) + (
                    longueur_d_onde - 633) / (705 - 633) * voltage_deg1(tableau_phase + alpha_705, alpha_705, beta_705)

    if 705 <= longueur_d_onde < 785:
        V = (785 - longueur_d_onde) / (785 - 705) * voltage_deg1(tableau_phase + alpha_705, alpha_705, beta_705) + (
                    longueur_d_onde - 705) / (785 - 705) * voltage_deg1(tableau_phase + alpha_785, alpha_785, beta_785)

    if longueur_d_onde >= 785:
        V = voltage_deg1(tableau_phase + alpha_785, alpha_785, beta_785)

    
    # Création de deux plots côte à côte : carte de phase et carte de voltage
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot de la carte de phase
    im1 = axes[0].imshow(carte_phase, cmap='gray_r', origin='lower', aspect='auto', vmin=0, vmax=2*np.pi)
    axes[0].set_title("Carte de Phase pour SLM")
    axes[0].set_xlabel("Pixels (X)")
    axes[0].set_ylabel("Pixels (Y)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Phase (radians)')

    # Plot de la carte de voltage
    im2 = axes[1].imshow(V, cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title("Carte de Voltage pour SLM")
    axes[1].set_xlabel("Pixels (X)")
    axes[1].set_ylabel("Pixels (Y)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Voltage (%)')

    # Ajustement de l'espacement
    plt.tight_layout(pad=2.0, w_pad=3.0)
    # Ou utilisez plt.subplots_adjust(wspace=0.3)

    # Affichage des plots
    plt.show()
    
    return V.astype('float64')


#appliquer la carte de voltage au SLM
error = slm.showData(carte_de_voltage((tableau_phase, 633)))
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

#on prend une photo après voltage :
time.sleep(0.1)
array_voltage = list(cam.take_raw_exposures(num_exposures=1))[0]
fits.writeto('D:\\slm_osae_2024\\resultats\\test_boite_noire\\Image_avecvoltage.fits', array_voltage, overwrite=True)