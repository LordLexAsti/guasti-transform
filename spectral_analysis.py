"""
TRANSFORMÉE DE GUASTI - PARTIE 5: ANALYSE SPECTRALE
====================================================

Ce script utilise la classe GuastiTransform pour construire un
signal global et rechercher des résonances spectrales.

Hypothèse: Les pics de résonance de ce signal correspondent
aux zéros non-triviaux de la fonction Zêta de Riemann.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lombscargle, find_peaks
from tqdm import tqdm
import warnings

# Importer la brique de base que tu as créée
try:
    from guasti_transform import GuastiTransform
except ImportError:
    print("Erreur: Le fichier 'guasti_transform.py' doit être dans le même dossier.")
    exit()

# Les 10 premiers zéros non-triviaux de Riemann (arrondis)
# C'est notre "vérité terrain" pour vérification
KNOWN_ZEROS = [
    14.1347, 21.0220, 25.0108, 30.4248, 32.9350,
    37.5861, 40.9187, 43.3270, 48.0051, 49.7738
]

def compute_guasti_signal(N_max=2000):
    """
    Construit le signal global de Guasti.
    
    Le "temps" (axe x) est log(n).
    Le "signal" (axe y) est la somme des poids de Möbius
    de la transformée de Guasti pour n.
    """
    print(f"Calcul du signal de Guasti (pondération Möbius) pour N_max = {N_max}...")
    
    # Éviter les avertissements de log(1) qui est 0
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    x_time = []  # Axe "temps" log(n)
    y_signal = [] # Axe "signal" M(n)
    
    # Nous utilisons tqdm pour une barre de progression, car c'est long
    for n in tqdm(range(1, N_max + 1)):
        # 1. Initialiser la transformée pour n avec la pondération Möbius
        gt = GuastiTransform(n, weight_function='mobius')
        
        # 2. Le signal pour ce n est la somme de tous les poids
        # C'est une nouvelle fonction M(n) = Σ_{i*j=n} μ(i)μ(j)
        signal_value = gt.weights.sum()
        
        # 3. Stocker le point (temps, signal)
        x_time.append(np.log(n))
        y_signal.append(signal_value)
        
    warnings.filterwarnings('default', category=RuntimeWarning)
    
    # Convertir en arrays numpy pour le calcul
    return np.array(x_time), np.array(y_signal)

def find_spectral_peaks(x_time, y_signal, max_freq=60, min_peak_height=0.01):
    """
    Calcule le périodogramme de Lomb-Scargle pour trouver les 
    fréquences de résonance.
    """
    print("Calcul du périodogramme de Lomb-Scargle (chasse aux zéros)...")
    
    # Définir les fréquences (t) que nous voulons tester.
    # Celles-ci correspondent à la partie imaginaire des zéros.
    # Nous testons 4000 points entre 0 et max_freq
    frequencies_to_test = np.linspace(0.1, max_freq, 4000)
    
    # Calculer le périodogramme. C'est le cœur de l'analyse!
    power_spectrum = lombscargle(x_time, y_signal, frequencies_to_test, normalize=True)
    
    # Trouver les pics dans notre spectre de puissance
    # 'height=min_peak_height' filtre le bruit de fond
    peak_indices, _ = find_peaks(power_spectrum, height=min_peak_height)
    
    detected_frequencies = frequencies_to_test[peak_indices]
    
    print(f"✓ {len(detected_frequencies)} pics spectraux détectés.")
    
    return frequencies_to_test, power_spectrum, detected_frequencies

def plot_spectrum_vs_zeros(frequencies, power, detected_peaks):
    """
    Visualise les résultats:
    Trace le spectre de puissance et le compare aux zéros de Riemann connus.
    """
    print("Génération du graphique spectral...")
    
    plt.figure(figsize=(16, 8))
    plt.plot(frequencies, power, label='Spectre de Puissance (Transformée de Guasti)', color='blue', alpha=0.7)
    
    # Marquer les pics que nous avons détectés
    peak_power = power[np.isin(frequencies, detected_peaks)]
    plt.plot(detected_peaks, peak_power, 'x', color='red', markersize=10, 
             label=f'Pics détectés ({len(detected_peaks)})')
    
    # Superposer les zéros de Riemann connus pour comparaison
    for i, zero in enumerate(KNOWN_ZEROS):
        if zero <= frequencies.max():
            plt.axvline(x=zero, color='green', linestyle='--', alpha=0.6,
                        label=f'Zéro de Riemann #{i+1} (Connu)' if i == 0 else '_nolegend_')
            
    plt.title(f'Analyse Spectrale de la Transformée de Guasti (N_max = {N_MAX})', fontsize=16)
    plt.xlabel('Fréquence (t) - Partie imaginaire des zéros de Riemann', fontsize=12)
    plt.ylabel('Puissance Spectrale (Normalisée)', fontsize=12)
    plt.legend()
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.xlim(0, frequencies.max())
    
    output_filename = 'guasti_spectral_analysis.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_filename}")
    plt.close()

def compare_results(detected_peaks):
    """
    Compare les pics détectés aux zéros connus et calcule la précision.
    """
    print("\n" + "="*80)
    print("COMPARAISON: PICS DÉTECTÉS vs ZÉROS DE RIEMANN CONNUS")
    print("="*80)
    print(f"{'Zéro Connu (t)':<15} | {'Pic Détecté (f)':<15} | {'Erreur |t - f|':<15}")
    print("-"*47)
    
    success_count = 0
    max_error_margin = 0.6 # C'est notre marge d'erreur (0.6 unité)
    
    for known_zero in KNOWN_ZEROS:
        # Trouver le pic détecté le plus proche
        if len(detected_peaks) > 0:
            errors = np.abs(detected_peaks - known_zero)
            best_match_index = np.argmin(errors)
            closest_peak = detected_peaks[best_match_index]
            error = errors[best_match_index]
            
            if error <= max_error_margin:
                print(f"{known_zero:<15.4f} | {closest_peak:<15.4f} | {error:<15.4f} ✓")
                success_count += 1
                # Enlever ce pic pour ne pas le réutiliser
                detected_peaks = np.delete(detected_peaks, best_match_index)
            else:
                print(f"{known_zero:<15.4f} | {'-':<15} | {'> 0.6':<15}")
        else:
            print(f"{known_zero:<15.4f} | {'-':<15} | {'-':<15}")
            
    print("-"*47)
    print(f"RÉSUMÉ: {success_count} / {len(KNOWN_ZEROS)} zéros détectés")
    print(f"         avec une marge d'erreur de {max_error_margin} unités.")
    print("="*80)

# ============================================================================
# MAIN: Exécution de l'analyse
# =================================_===========================================

if __name__ == "__main__":
    
    # Augmenter N_max améliore la précision mais prend plus de temps
    # 2000 est un bon test. 5000+ est mieux.
    N_MAX = 2500
    
    # 1. Construire le signal à partir de ta classe
    x_time, y_signal = compute_guasti_signal(N_max=N_MAX)
    
    # 2. Trouver les pics spectraux
    # Nous cherchons jusqu'à t=50 (pour couvrir les 10 premiers zéros)
    freqs, power, peaks = find_spectral_peaks(x_time, y_signal, max_freq=55, min_peak_height=0.015)
    
    # 3. Afficher les résultats
    plot_spectrum_vs_zeros(freqs, power, peaks)
    
    # 4. Comparer et valider
    compare_results(peaks)