"""
PÉRIODOGRAMME DE DELTA(x) - Analyse spectrale de la Grille de Guasti
========================================================================

Ce script cherche les "fréquences" cachées dans les oscillations de Δ(x).
Si l'Hypothèse de Riemann est vraie, ces fréquences devraient correspondre
aux parties imaginaires des zéros de ζ(s).

Pour débutants Python: Chaque étape est expliquée en détail !
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# ============================================================================
# ÉTAPE 1: Recalculer Δ(x) avec plus de précision
# ============================================================================

def sieve_divisor_counts(N: int):
    """
    Compte combien de diviseurs a chaque nombre jusqu'à N.
    C'est la fonction d(n) en théorie des nombres.
    """
    d = [0]*(N+1)
    for i in range(1, N+1):
        # Chaque multiple de i a i comme diviseur
        for j in range(i, N+1, i):
            d[j] += 1
    return d

def compute_delta(N: int):
    """
    Calcule Δ(x) = D(x) - x·log(x) - (2γ-1)x
    
    D(x) = somme cumulée des diviseurs = comptage hyperbolique sur la grille
    Le terme x·log(x) + (2γ-1)x est la "prédiction lisse"
    Δ(x) = les oscillations résiduelles = signature des zéros de ζ
    """
    print(f"Calcul de d(n) pour n=1 à {N}...")
    d = sieve_divisor_counts(N)
    
    print("Calcul de la somme cumulée D(x)...")
    D = np.cumsum(d[1:])  # D(x) = Σ d(n) pour n≤x
    
    print("Calcul de Δ(x)...")
    x = np.arange(1, N+1, dtype=float)
    gamma = 0.577215664901532860606512090082402431  # constante d'Euler-Mascheroni
    
    # Le terme principal prédit par la théorie analytique
    main_term = x * np.log(x) + (2*gamma - 1.0) * x
    
    # Les oscillations résiduelles
    Delta = D - main_term
    
    return x, Delta

# ============================================================================
# ÉTAPE 2: Transformée de Fourier pour trouver les fréquences
# ============================================================================

def compute_periodogram(x, Delta, log_scale=True):
    """
    Calcule le périodogramme = spectre de puissance des oscillations.
    
    En utilisant log(x) comme variable, car les zéros apparaissent
    dans les oscillations de type x^(1/2 + it) où t = Im(ρ).
    """
    if log_scale:
        print("Transformation en échelle logarithmique...")
        # On travaille en log(x) car les oscillations ont la forme x^ρ
        log_x = np.log(x)
        
        # Interpolation régulière en log(x)
        log_x_uniform = np.linspace(log_x[0], log_x[-1], len(x))
        Delta_uniform = np.interp(log_x_uniform, log_x, Delta)
        
        # Fenêtrage pour réduire les effets de bord
        window = signal.windows.hann(len(Delta_uniform))
        Delta_windowed = Delta_uniform * window
        
        # Transformée de Fourier
        print("Calcul de la transformée de Fourier...")
        spectrum = fft(Delta_windowed)
        freqs = fftfreq(len(Delta_windowed), d=(log_x_uniform[1] - log_x_uniform[0]))
        
        # On garde seulement les fréquences positives
        positive_freqs = freqs > 0
        freqs = freqs[positive_freqs]
        power = np.abs(spectrum[positive_freqs])**2
        
    else:
        # FFT directe sur x (moins adaptée pour les zéros de Riemann)
        window = signal.windows.hann(len(Delta))
        Delta_windowed = Delta * window
        
        spectrum = fft(Delta_windowed)
        freqs = fftfreq(len(Delta_windowed), d=(x[1] - x[0]))
        
        positive_freqs = freqs > 0
        freqs = freqs[positive_freqs]
        power = np.abs(spectrum[positive_freqs])**2
    
    return freqs, power

# ============================================================================
# ÉTAPE 3: Les zéros de Riemann connus (pour comparaison)
# ============================================================================

# Parties imaginaires des premiers zéros non-triviaux de ζ(s)
# Source: tables de calculs haute précision
RIEMANN_ZEROS = [
    14.134725,  # Premier zéro
    21.022040,
    25.010858,
    30.424876,
    32.935062,
    37.586178,
    40.918719,
    43.327073,
    48.005151,
    49.773832,
    52.970321,
    56.446248,
    59.347044,
    60.831779,
    65.112544,
    67.079811,
    69.546402,
    72.067158,
    75.704691,
    77.144840,
    79.337375,
    82.910381,
    84.735493,
    87.425275,
    88.809111,
    92.491899,
    94.651344,
    95.870634,
    98.831194,
    101.317851,
]

# ============================================================================
# ÉTAPE 4: Visualisation du périodogramme
# ============================================================================

def plot_periodogram_with_zeros(freqs, power, riemann_zeros, max_freq=120):
    """
    Trace le périodogramme et marque les positions des zéros de Riemann.
    """
    # Limiter à la plage d'intérêt
    mask = freqs <= max_freq
    freqs_plot = freqs[mask]
    power_plot = power[mask]
    
    # Normalisation pour la visualisation
    power_plot = power_plot / np.max(power_plot)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ---- Graphique 1: Vue complète ----
    ax1.semilogy(freqs_plot, power_plot, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Fréquence (≈ partie imaginaire des zéros)", fontsize=12)
    ax1.set_ylabel("Puissance spectrale (normalisée)", fontsize=12)
    ax1.set_title("Périodogramme de Δ(x) - Vue complète", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Marquer les zéros de Riemann connus
    for zero in riemann_zeros:
        if zero <= max_freq:
            ax1.axvline(zero, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Légende
    ax1.axvline(riemann_zeros[0], color='red', linestyle='--', alpha=0.6, 
                linewidth=1, label='Zéros de Riemann connus')
    ax1.legend(fontsize=10)
    
    # ---- Graphique 2: Zoom sur les premiers zéros ----
    zoom_mask = freqs <= 50
    ax2.plot(freqs[zoom_mask], power[zoom_mask], linewidth=0.8)
    ax2.set_xlabel("Fréquence", fontsize=12)
    ax2.set_ylabel("Puissance spectrale", fontsize=12)
    ax2.set_title("Zoom sur les 10 premiers zéros de Riemann", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Marquer les 10 premiers zéros avec leurs valeurs
    for i, zero in enumerate(riemann_zeros[:10]):
        ax2.axvline(zero, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        if i % 2 == 0:  # Alterner les labels pour éviter le chevauchement
            ax2.text(zero, ax2.get_ylim()[1]*0.9, f'{zero:.2f}', 
                    rotation=90, va='top', ha='right', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/periodogram_riemann_zeros.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Graphique sauvegardé: periodogram_riemann_zeros.png")

# ============================================================================
# ÉTAPE 5: Détection des pics et comparaison avec les zéros
# ============================================================================

def find_peaks_and_compare(freqs, power, riemann_zeros, num_peaks=15):
    """
    Trouve les pics les plus importants dans le périodogramme
    et les compare avec les zéros de Riemann connus.
    """
    # Trouver les pics locaux
    peaks, properties = signal.find_peaks(power, prominence=np.max(power)*0.01)
    
    # Trier par puissance décroissante
    peak_powers = power[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1]
    top_peaks = peaks[sorted_indices[:num_peaks]]
    top_freqs = freqs[top_peaks]
    top_powers = power[top_peaks]
    
    # Créer un tableau de comparaison
    results = []
    for i, (freq, pwr) in enumerate(zip(top_freqs, top_powers)):
        # Trouver le zéro de Riemann le plus proche
        distances = [abs(freq - zero) for zero in riemann_zeros]
        min_dist = min(distances)
        closest_zero = riemann_zeros[distances.index(min_dist)]
        
        results.append({
            'Rang': i+1,
            'Fréquence détectée': f'{freq:.3f}',
            'Puissance': f'{pwr:.2e}',
            'Zéro le plus proche': f'{closest_zero:.3f}',
            'Écart': f'{min_dist:.3f}',
            'Match': '✓✓✓' if min_dist < 0.5 else '✓✓' if min_dist < 1.0 else '✓' if min_dist < 2.0 else '?'
        })
    
    df = pd.DataFrame(results)
    return df

# ============================================================================
# MAIN: Exécution de l'analyse complète
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ANALYSE SPECTRALE DE LA GRILLE DE GUASTI")
    print("Recherche des zéros de Riemann dans Δ(x)")
    print("="*80)
    print()
    
    # Paramètres
    N = 50000  # Augmenté pour plus de précision
    
    # 1. Calculer Δ(x)
    print(f"[1/4] Calcul de Δ(x) jusqu'à N={N}...")
    x, Delta = compute_delta(N)
    print(f"      → Amplitude de Δ(x): [{Delta.min():.2f}, {Delta.max():.2f}]")
    print()
    
    # 2. Périodogramme
    print("[2/4] Calcul du périodogramme en échelle logarithmique...")
    freqs, power = compute_periodogram(x, Delta, log_scale=True)
    print(f"      → {len(freqs)} fréquences analysées")
    print(f"      → Plage: [0, {freqs.max():.2f}]")
    print()
    
    # 3. Visualisation
    print("[3/4] Création des graphiques...")
    plot_periodogram_with_zeros(freqs, power, RIEMANN_ZEROS, max_freq=120)
    print()
    
    # 4. Détection et comparaison
    print("[4/4] Détection des pics et comparaison avec les zéros de Riemann...")
    comparison_df = find_peaks_and_compare(freqs, power, RIEMANN_ZEROS, num_peaks=20)
    print()
    print("="*80)
    print("RÉSULTATS: TOP 20 DES PICS DÉTECTÉS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print()
    print("Légende:")
    print("  ✓✓✓ = Excellent match (écart < 0.5)")
    print("  ✓✓  = Bon match (écart < 1.0)")
    print("  ✓   = Match acceptable (écart < 2.0)")
    print("  ?   = Pas de correspondance claire")
    print()
    print("="*80)
    print("ANALYSE TERMINÉE !")
    print("="*80)
    
    # Sauvegarder les résultats dans un CSV
    comparison_df.to_csv('/mnt/user-data/outputs/periodogram_peaks_comparison.csv', 
                         index=False)
    print("\n✓ Tableau sauvegardé: periodogram_peaks_comparison.csv")
