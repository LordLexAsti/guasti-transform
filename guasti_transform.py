"""
TRANSFORMÉE DE GUASTI - Implémentation formelle
================================================

La Transformée de Guasti décompose la structure multiplicative d'un nombre
en sa "signature angulaire" sur la grille.

Pour débutants Python: Tout est expliqué pas à pas!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from math import atan2, pi, sqrt, log, gcd

# ============================================================================
# PARTIE 1: Définition de la Transformée de Guasti
# ============================================================================

class GuastiTransform:
    """
    Classe pour calculer et analyser la Transformée de Guasti d'un nombre.
    
    La transformée encode la structure multiplicative d'un nombre via
    les angles de ses factorisations sur la grille.
    """
    
    def __init__(self, n, weight_function='uniform'):
        """
        Initialise la transformée pour un nombre n.
        
        Paramètres:
        -----------
        n : int
            Le nombre à analyser
        weight_function : str
            Type de pondération:
            - 'uniform': tous les diviseurs comptent pareil (défaut)
            - 'mobius': pondération par μ(i)×μ(j)
            - 'log': pondération logarithmique log(i)×log(j)
            - 'balanced': pondération symétrique sqrt(i/j) + sqrt(j/i)
        """
        self.n = n
        self.weight_function = weight_function
        
        # Trouver toutes les factorisations i×j = n
        self.factorizations = []
        self.angles = []
        self.weights = []
        
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0:
                j = n // i
                
                # Angle en radians
                theta = atan2(j, i)
                
                # Poids selon la fonction choisie
                w = self._compute_weight(i, j)
                
                # Stocker la factorisation
                self.factorizations.append((i, j))
                self.angles.append(theta)
                self.weights.append(w)
                
                # Si i ≠ j, ajouter aussi la factorisation symétrique
                if i != j:
                    theta_sym = atan2(i, j)
                    w_sym = self._compute_weight(j, i)
                    self.factorizations.append((j, i))
                    self.angles.append(theta_sym)
                    self.weights.append(w_sym)
        
        self.angles = np.array(self.angles)
        self.weights = np.array(self.weights)
        
        # Nombre de diviseurs
        self.tau = len(self.factorizations)
    
    def _compute_weight(self, i, j):
        """Calcule le poids w(i,j) selon la fonction choisie"""
        if self.weight_function == 'uniform':
            return 1.0
        
        elif self.weight_function == 'mobius':
            return self._mobius(i) * self._mobius(j)
        
        elif self.weight_function == 'log':
            return log(i) * log(j) if i > 1 and j > 1 else 0.0
        
        elif self.weight_function == 'balanced':
            return sqrt(i/j) + sqrt(j/i)
        
        else:
            return 1.0
    
    @staticmethod
    def _mobius(n):
        """Fonction de Möbius μ(n)"""
        if n == 1:
            return 1
        
        # Factorisation simple pour calculer μ
        factors = []
        temp = n
        d = 2
        while d * d <= temp:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count > 1:  # Facteur carré
                return 0
            if count == 1:
                factors.append(d)
            d += 1
        
        if temp > 1:
            factors.append(temp)
        
        return (-1) ** len(factors)
    
    def get_spectrum(self, num_bins=360):
        """
        Retourne le spectre angulaire discret.
        
        Divise [0, π/2] en num_bins intervalles et compte
        les contributions dans chaque bin.
        """
        bins = np.linspace(0, pi/2, num_bins + 1)
        spectrum = np.zeros(num_bins)
        
        for angle, weight in zip(self.angles, self.weights):
            # Trouver le bin correspondant
            bin_idx = np.searchsorted(bins[:-1], angle, side='right') - 1
            if 0 <= bin_idx < num_bins:
                spectrum[bin_idx] += weight
        
        # Centres des bins pour le plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return bin_centers, spectrum
    
    def entropy(self):
        """
        Calcule l'entropie de Shannon de la distribution angulaire.
        
        Mesure la "complexité" de la structure multiplicative:
        - Entropie faible = peu de factorisations (nombres premiers)
        - Entropie élevée = beaucoup de factorisations (nombres hautement composés)
        """
        if len(self.weights) == 0:
            return 0.0
        
        # Normaliser les poids pour avoir une distribution de probabilité
        probs = np.abs(self.weights) / np.sum(np.abs(self.weights))
        probs = probs[probs > 0]  # Enlever les zéros
        
        return -np.sum(probs * np.log2(probs))
    
    def is_prime_signature(self):
        """
        Teste si la signature angulaire correspond à un nombre premier.
        
        Un premier a exactement 2 factorisations: n×1 et 1×n
        """
        return self.tau == 2

# ============================================================================
# PARTIE 2: Visualisation de la Transformée
# ============================================================================

def plot_guasti_transform(numbers, weight='uniform'):
    """
    Visualise la Transformée de Guasti pour plusieurs nombres.
    
    Compare les signatures angulaires de nombres premiers,
    composés simples, et hautement composés.
    """
    n_numbers = len(numbers)
    fig = plt.figure(figsize=(16, 4 * ((n_numbers + 1) // 2)))
    gs = GridSpec(((n_numbers + 1) // 2), 2, figure=fig, hspace=0.3, wspace=0.3)
    
    results = []
    
    for idx, n in enumerate(numbers):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col], projection='polar')
        
        # Calculer la transformée
        gt = GuastiTransform(n, weight_function=weight)
        
        # Afficher en coordonnées polaires
        for angle, w in zip(gt.angles, gt.weights):
            # Hauteur proportionnelle au poids
            r = abs(w)
            color = 'blue' if w > 0 else 'red'
            ax.plot([angle, angle], [0, r], color=color, linewidth=2, alpha=0.7)
            ax.scatter([angle], [r], color=color, s=50, zorder=5)
        
        # Configuration
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_ylim(0, max(np.abs(gt.weights)) * 1.2 if len(gt.weights) > 0 else 1)
        
        # Titre avec infos
        prime_str = " (PREMIER)" if gt.is_prime_signature() else ""
        ax.set_title(f'n = {n}{prime_str}\nτ(n) = {gt.tau}, H = {gt.entropy():.2f}',
                    fontsize=11, fontweight='bold')
        
        # Stocker les résultats
        results.append({
            'n': n,
            'τ(n)': gt.tau,
            'Entropie': f'{gt.entropy():.3f}',
            'Premier?': '✓' if gt.is_prime_signature() else '✗',
            'Factorisations': str(gt.factorizations[:3]) + ('...' if gt.tau > 6 else '')
        })
    
    plt.savefig(f'/mnt/user-data/outputs/guasti_transform_{weight}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return pd.DataFrame(results)

# ============================================================================
# PARTIE 3: Analyse comparative
# ============================================================================

def analyze_number_classes():
    """
    Analyse comparative de différentes classes de nombres:
    - Nombres premiers
    - Nombres hautement composés
    - Puissances de premiers
    - Nombres avec facteurs variés
    """
    
    print("="*80)
    print("ANALYSE PAR CLASSE DE NOMBRES")
    print("="*80)
    print()
    
    # Différentes classes
    test_sets = {
        'Premiers': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        'Puissances de 2': [2, 4, 8, 16, 32, 64, 128, 256],
        'Hautement composés': [1, 2, 4, 6, 12, 24, 36, 48, 60, 120],
        'Carrés parfaits': [4, 9, 16, 25, 36, 49, 64, 81, 100],
    }
    
    results = {}
    
    for class_name, numbers in test_sets.items():
        print(f"\n{class_name}:")
        print("-" * 40)
        
        entropies = []
        taus = []
        
        for n in numbers:
            gt = GuastiTransform(n)
            entropies.append(gt.entropy())
            taus.append(gt.tau)
            print(f"  n={n:3d}: τ(n)={gt.tau:3d}, H={gt.entropy():.3f}")
        
        results[class_name] = {
            'Entropie moyenne': np.mean(entropies),
            'τ(n) moyen': np.mean(taus),
            'Complexité': 'Simple' if np.mean(entropies) < 2 else 'Modérée' if np.mean(entropies) < 4 else 'Élevée'
        }
    
    print("\n" + "="*80)
    print("RÉSUMÉ COMPARATIF")
    print("="*80)
    df = pd.DataFrame(results).T
    print(df.to_string())
    print()
    
    return df

# ============================================================================
# PARTIE 4: Transformée de Guasti pour grands nombres
# ============================================================================

def large_number_analysis(n):
    """
    Analyse détaillée de la structure multiplicative d'un grand nombre.
    
    Utile pour comprendre la divisibilité des nombres composés.
    """
    print(f"\nANALYSE DÉTAILLÉE: n = {n}")
    print("="*60)
    
    gt = GuastiTransform(n, weight_function='uniform')
    
    print(f"\nPropriétés de base:")
    print(f"  • Nombre de diviseurs: τ({n}) = {gt.tau}")
    print(f"  • Entropie de structure: H = {gt.entropy():.4f}")
    print(f"  • Est premier? {gt.is_prime_signature()}")
    
    print(f"\nFactorisations (i × j = {n}):")
    for i, (i_val, j_val) in enumerate(gt.factorizations[:10]):
        angle_deg = gt.angles[i] * 180 / pi
        print(f"  {i_val:6d} × {j_val:6d} = {n:8d}  →  θ = {angle_deg:6.2f}°")
    
    if gt.tau > 10:
        print(f"  ... et {gt.tau - 10} autres factorisations")
    
    # Spectre angulaire
    bin_centers, spectrum = gt.get_spectrum(num_bins=90)
    
    print(f"\nDistribution angulaire:")
    print(f"  • Régions avec le plus de factorisations:")
    top_bins = np.argsort(spectrum)[-5:][::-1]
    for rank, bin_idx in enumerate(top_bins[:3]):
        angle_deg = bin_centers[bin_idx] * 180 / pi
        count = spectrum[bin_idx]
        if count > 0:
            print(f"    {rank+1}. Autour de {angle_deg:.1f}°: {int(count)} factorisations")
    
    print()
    return gt

# ============================================================================
# MAIN: Démonstration
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TRANSFORMÉE DE GUASTI - Analyse de la structure multiplicative")
    print("="*80)
    print()
    
    print("[1/4] Visualisation de nombres représentatifs...")
    numbers_to_visualize = [
        2,      # Premier
        6,      # Premier nombre composé
        12,     # Hautement composé
        17,     # Premier
        30,     # Premier nombre avec 3 facteurs premiers distincts
        60,     # Hautement composé
        127,    # Premier de Mersenne
        128,    # Puissance de 2
    ]
    
    df_viz = plot_guasti_transform(numbers_to_visualize, weight='uniform')
    print("✓ Graphique sauvegardé: guasti_transform_uniform.png")
    print("\nTableau récapitulatif:")
    print(df_viz.to_string(index=False))
    print()
    
    print("\n[2/4] Analyse par classes de nombres...")
    df_classes = analyze_number_classes()
    df_classes.to_csv('/mnt/user-data/outputs/guasti_transform_classes.csv')
    print("✓ Résultats sauvegardés: guasti_transform_classes.csv")
    
    print("\n[3/4] Analyse de grands nombres composés...")
    large_numbers = [
        120,    # Premier nombre avec τ(n) = 16
        360,    # Nombre hautement composé
        1260,   # 2² × 3² × 5 × 7
        2520,   # Plus petit nombre divisible par 1..10
    ]
    
    for n in large_numbers:
        large_number_analysis(n)
    
    print("\n[4/4] Transformée avec pondération Möbius...")
    df_mobius = plot_guasti_transform([6, 12, 30, 60], weight='mobius')
    print("✓ Graphique sauvegardé: guasti_transform_mobius.png")
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE!")
    print("="*80)
    print("\nLa Transformée de Guasti révèle:")
    print("  • Les nombres PREMIERS ont une signature minimale (2 pics à 0° et 90°)")
    print("  • Les nombres COMPOSÉS ont une signature riche avec multiples angles")
    print("  • L'ENTROPIE mesure la complexité de la structure multiplicative")
    print("  • La distribution angulaire encode la DIVISIBILITÉ")
    print("="*80)
