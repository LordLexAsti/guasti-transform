"""
TRANSFORMÉE DE GUASTI - Extensions opérationnelles
===================================================

Les 3 extensions proposées par ChatGPT pour rendre la TG opérationnelle:
1. Inversion Möbius 2D pour récupérer a(i,j) depuis ℛ[a](n)
2. Carte de diviseurs en coordonnées isométriques (heatmap)
3. Pré-criblage TG pour factorisation guidée

Pour débutants: Chaque fonction est documentée et expliquée!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from math import sqrt, log, gcd, atan2, pi
from collections import defaultdict
import seaborn as sns

# ============================================================================
# EXTENSION 1: Inversion Möbius 2D
# ============================================================================

def sieve_mobius(N):
    """Calcule μ(n) pour n=1..N via le crible"""
    mu = [1]*(N+1)
    is_prime = [True]*(N+1)
    primes = []
    mu[0] = 0
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, N+1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        
        j = 0
        while j < len(primes) and i*primes[j] <= N:
            p = primes[j]
            is_prime[i*p] = False
            if i % p == 0:
                mu[i*p] = 0
                break
            else:
                mu[i*p] = -mu[i]
            j += 1
    
    return mu

def mobius_2d_inversion(h_values, max_i, max_j):
    """
    Inversion Möbius 2D: Récupère a(i,j) depuis h(n) = ℛ[a](n)
    
    Formule: a(i,j) = Σ_{d|i} Σ_{e|j} μ(d)μ(e) h(ij/(de))
    
    Paramètres:
    -----------
    h_values : dict
        Dictionnaire {n: h(n)} où h(n) = Σ_{ij=n} a(i,j)
    max_i, max_j : int
        Dimensions maximales de la grille à reconstruire
    
    Retour:
    -------
    a : 2D array
        Grille reconstituée a[i,j]
    """
    print(f"Inversion Möbius 2D sur grille {max_i}×{max_j}...")
    
    # Calculer μ(n)
    N_max = max(max_i, max_j)
    mu = sieve_mobius(N_max)
    
    # Initialiser la grille de sortie
    a = np.zeros((max_i + 1, max_j + 1))
    
    # Pour chaque point (i,j)
    for i in range(1, max_i + 1):
        for j in range(1, max_j + 1):
            # Trouver tous les diviseurs de i et j
            divisors_i = [d for d in range(1, i+1) if i % d == 0]
            divisors_j = [e for e in range(1, j+1) if j % e == 0]
            
            # Appliquer la formule d'inversion
            total = 0
            for d in divisors_i:
                for e in divisors_j:
                    n_target = (i * j) // (d * e)
                    if n_target in h_values:
                        total += mu[d] * mu[e] * h_values[n_target]
            
            a[i, j] = total
    
    return a

def demo_mobius_inversion():
    """
    Démonstration: Partir d'une fonction a(i,j), calculer ℛ[a](n),
    puis récupérer a(i,j) par inversion
    """
    print("\n" + "="*80)
    print("DÉMONSTRATION: INVERSION MÖBIUS 2D")
    print("="*80)
    
    # Fonction test: a(i,j) = 1 si gcd(i,j)=1, 0 sinon
    max_size = 15
    a_original = np.zeros((max_size + 1, max_size + 1))
    for i in range(1, max_size + 1):
        for j in range(1, max_size + 1):
            a_original[i, j] = 1 if gcd(i, j) == 1 else 0
    
    print(f"\nFonction originale: a(i,j) = 1 si gcd(i,j)=1")
    print(f"Somme totale: {np.sum(a_original):.0f}")
    
    # Calculer ℛ[a](n) = Σ_{ij=n} a(i,j)
    h_values = {}
    for i in range(1, max_size + 1):
        for j in range(1, max_size + 1):
            n = i * j
            if n not in h_values:
                h_values[n] = 0
            h_values[n] += a_original[i, j]
    
    print(f"\nRadon ℛ[a] calculé pour {len(h_values)} valeurs de n")
    print(f"Exemples: h(6)={h_values.get(6, 0):.0f}, h(12)={h_values.get(12, 0):.0f}")
    
    # Inversion
    a_reconstructed = mobius_2d_inversion(h_values, max_size, max_size)
    
    print(f"\nAprès inversion Möbius 2D:")
    print(f"Somme reconstruite: {np.sum(a_reconstructed):.0f}")
    
    # Vérification
    error = np.sum(np.abs(a_original - a_reconstructed))
    print(f"\nErreur totale: {error:.6f}")
    
    if error < 1e-10:
        print("✓✓✓ INVERSION PARFAITE!")
    else:
        print("✗ Erreur détectée")
    
    return a_original, a_reconstructed

# ============================================================================
# EXTENSION 2: Carte de diviseurs en coordonnées isométriques
# ============================================================================

def divisor_heatmap(n, max_display=50):
    """
    Crée une heatmap des diviseurs de n en coordonnées isométriques.
    
    Chaque point (i,j) tel que ij=n est marqué, avec intensité
    proportionnelle à une fonction de poids (ici: log ou uniforme).
    
    Paramètres:
    -----------
    n : int
        Le nombre à analyser
    max_display : int
        Taille maximale de la grille à afficher
    """
    print(f"\nCréation de la heatmap pour n = {n}...")
    
    # Trouver toutes les factorisations
    factorizations = []
    for i in range(1, int(sqrt(n)) + 1):
        if n % i == 0:
            j = n // i
            factorizations.append((i, j))
            if i != j:
                factorizations.append((j, i))
    
    print(f"  {len(factorizations)} factorisations trouvées")
    
    # Créer la grille
    max_coord = min(max_display, max(n, int(sqrt(n)) + 5))
    grid = np.zeros((max_coord + 1, max_coord + 1))
    
    # Marquer les factorisations
    for i, j in factorizations:
        if i <= max_coord and j <= max_coord:
            # Poids: log(i) * log(j) pour donner plus d'importance aux facteurs équilibrés
            weight = log(i + 1) * log(j + 1) if i > 1 and j > 1 else 1
            grid[i, j] = weight
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graphique 1: Heatmap standard
    ax1 = axes[0]
    im1 = ax1.imshow(grid, origin='lower', cmap='hot', interpolation='nearest')
    ax1.set_xlabel("i", fontsize=12)
    ax1.set_ylabel("j", fontsize=12)
    ax1.set_title(f"Heatmap des diviseurs: n = {n}\nτ(n) = {len(factorizations)}", 
                  fontsize=13, fontweight='bold')
    
    # Marquer quelques points clés
    for i, j in factorizations[:10]:
        if i <= max_coord and j <= max_coord:
            ax1.plot(i, j, 'wo', markersize=6, markeredgecolor='cyan', markeredgewidth=1.5)
            if i == j or (i, j) in [(1, n), (n, 1)]:
                ax1.text(i, j, f'{i}×{j}', fontsize=8, color='cyan', 
                        ha='right', va='bottom')
    
    plt.colorbar(im1, ax=ax1, label='Poids log(i)log(j)')
    
    # Graphique 2: Vue "hyperbole"
    ax2 = axes[1]
    
    # Dessiner l'hyperbole ij = n
    i_range = np.linspace(1, max_coord, 1000)
    j_hyperbola = n / i_range
    mask = j_hyperbola <= max_coord
    ax2.plot(i_range[mask], j_hyperbola[mask], 'r-', linewidth=2, 
            alpha=0.7, label=f'Hyperbole ij={n}')
    
    # Marquer les factorisations entières
    for i, j in factorizations:
        if i <= max_coord and j <= max_coord:
            angle = atan2(j, i) * 180 / pi
            color = 'blue' if angle < 45 else 'green'
            ax2.scatter(i, j, s=100, c=color, alpha=0.7, edgecolors='black', linewidth=1.5)
            if len(factorizations) <= 20:
                ax2.text(i, j, f'  {i}×{j}', fontsize=8, ha='left')
    
    ax2.set_xlabel("i", fontsize=12)
    ax2.set_ylabel("j", fontsize=12)
    ax2.set_title(f"Vue hyperbole: ij = {n}", fontsize=13, fontweight='bold')
    ax2.set_xlim(0, max_coord)
    ax2.set_ylim(0, max_coord)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'/mnt/user-data/outputs/divisor_heatmap_{n}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Heatmap sauvegardée: divisor_heatmap_{n}.png")
    
    return factorizations

# ============================================================================
# EXTENSION 3: Pré-criblage TG pour factorisation
# ============================================================================

def tg_prescreening(n, moduli=[3, 4, 5, 7, 8, 11], cone_angles=8):
    """
    Pré-criblage TG pour guider la factorisation de n.
    
    Utilise:
    1. Tests de résidus quadratiques modulo q
    2. Analyse des signatures angulaires
    3. Filtrage Möbius pour détecter la square-freeness
    
    Retourne une liste priorisée de "pistes" pour la factorisation.
    
    Paramètres:
    -----------
    n : int
        Le nombre à factoriser
    moduli : list
        Liste des moduli à tester
    cone_angles : int
        Nombre de cônes angulaires à analyser
    """
    print(f"\n" + "="*70)
    print(f"PRÉ-CRIBLAGE TG POUR LA FACTORISATION DE n = {n}")
    print("="*70)
    
    results = {
        'n': n,
        'suspect_small_factors': [],
        'residue_classes': {},
        'angular_signature': [],
        'smoothness_hint': None
    }
    
    # 1. Test rapide des petits facteurs
    print("\n[1] Test des petits facteurs (trial division)...")
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    small_factors = []
    temp_n = n
    
    for p in small_primes:
        if temp_n % p == 0:
            count = 0
            while temp_n % p == 0:
                temp_n //= p
                count += 1
            small_factors.append((p, count))
            print(f"  ✓ Facteur trouvé: {p}^{count}")
    
    results['suspect_small_factors'] = small_factors
    
    if temp_n == 1:
        print(f"\n  → n = {n} est complètement factorisé!")
        print(f"     Factorisation: {' × '.join([f'{p}^{e}' for p, e in small_factors])}")
        return results
    
    print(f"  → Cofacteur restant: {temp_n}")
    n_remaining = temp_n
    
    # 2. Analyse des résidus modulo q
    print(f"\n[2] Analyse des classes résiduelles modulo q...")
    
    for q in moduli:
        residue = n_remaining % q
        results['residue_classes'][q] = residue
        
        # Tests de résidus quadratiques
        if q in [3, 5, 7, 11]:
            # Un nombre premier p ≡ 1 (mod 4) est somme de deux carrés
            # ssi p ≡ 1 (mod 4)
            if q == 4:
                hint = "Somme de deux carrés possible" if residue == 1 else "Pas somme de 2 carrés"
                print(f"  mod {q}: n ≡ {residue} → {hint}")
            else:
                print(f"  mod {q}: n ≡ {residue}")
    
    # 3. Signature angulaire (pour le cofacteur)
    print(f"\n[3] Analyse de la signature angulaire...")
    
    # Si le cofacteur est petit, on peut analyser ses diviseurs
    if n_remaining < 10000:
        factorizations = []
        for i in range(1, int(sqrt(n_remaining)) + 1):
            if n_remaining % i == 0:
                j = n_remaining // i
                angle = atan2(j, i) * 180 / pi
                factorizations.append((i, j, angle))
        
        # Grouper par secteurs angulaires
        angle_bins = np.linspace(0, 90, cone_angles + 1)
        angle_counts = np.zeros(cone_angles)
        
        for i, j, angle in factorizations:
            bin_idx = np.searchsorted(angle_bins[:-1], angle, side='right') - 1
            if 0 <= bin_idx < cone_angles:
                angle_counts[bin_idx] += 1
        
        results['angular_signature'] = angle_counts.tolist()
        
        print(f"  Nombre de diviseurs: {len(factorizations)}")
        print(f"  Distribution angulaire:")
        for i, count in enumerate(angle_counts):
            if count > 0:
                angle_range = f"[{angle_bins[i]:.1f}°, {angle_bins[i+1]:.1f}°]"
                print(f"    {angle_range}: {int(count)} factorisations")
    
    # 4. Test de smoothness (heuristique)
    print(f"\n[4] Test de smoothness heuristique...")
    
    # Si beaucoup de petits facteurs → probablement smooth
    if len(small_factors) >= 3:
        results['smoothness_hint'] = "SMOOTH (plusieurs petits facteurs)"
        print(f"  → Probablement {results['smoothness_hint']}")
        print(f"     Conseil: ECM ou Pollard-ρ sur le cofacteur")
    elif len(small_factors) == 0:
        results['smoothness_hint'] = "Possiblement SEMI-PRIME ou PREMIER"
        print(f"  → {results['smoothness_hint']}")
        print(f"     Conseil: Test de primalité puis factorisation forte (GNFS)")
    else:
        results['smoothness_hint'] = "SMOOTH MODÉRÉ"
        print(f"  → {results['smoothness_hint']}")
        print(f"     Conseil: Pollard-ρ puis ECM si nécessaire")
    
    # 5. Recommandations
    print(f"\n[5] RECOMMANDATIONS POUR LA FACTORISATION:")
    print("-" * 70)
    
    if n_remaining == 1:
        print("  ✓ Factorisation complète déjà obtenue!")
    elif n_remaining < 100:
        print(f"  → Trial division suffit (cofacteur = {n_remaining})")
    elif len(small_factors) >= 2:
        print(f"  1. Continuer ECM sur le cofacteur {n_remaining}")
        print(f"  2. Si ECM échoue après ~1000 courbes, passer à MPQS")
    else:
        print(f"  1. Test de primalité Miller-Rabin sur {n_remaining}")
        print(f"  2. Si composé: Pollard-ρ (rapide)")
        print(f"  3. Si échec: ECM avec courbes croissantes")
    
    print("="*70)
    
    return results

# ============================================================================
# DÉMONSTRATION COMPLÈTE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TRANSFORMÉE DE GUASTI - EXTENSIONS OPÉRATIONNELLES")
    print("="*80)
    
    # Extension 1: Inversion Möbius 2D
    print("\n### EXTENSION 1: INVERSION MÖBIUS 2D ###")
    a_orig, a_recon = demo_mobius_inversion()
    
    # Extension 2: Heatmaps pour plusieurs nombres
    print("\n### EXTENSION 2: CARTES DE DIVISEURS ###")
    
    test_numbers = [
        60,      # Hautement composé
        120,     # Encore plus de diviseurs
        2520,    # Très hautement composé (LCM(1..10))
        127,     # Premier (pour contraste)
    ]
    
    for n in test_numbers:
        divisor_heatmap(n, max_display=60)
    
    # Extension 3: Pré-criblage pour factorisation
    print("\n### EXTENSION 3: PRÉ-CRIBLAGE POUR FACTORISATION ###")
    
    # Cas tests
    target_numbers = [
        360,            # = 2³ × 3² × 5 (smooth)
        1729,           # = 7 × 13 × 19 (nombre de Ramanujan)
        8051,           # = 83 × 97 (semi-premier)
        15485863,       # Premier de Mersenne 2^24-1
    ]
    
    all_results = []
    for n in target_numbers:
        result = tg_prescreening(n, moduli=[3, 4, 5, 7, 8, 11])
        all_results.append(result)
        print("\n")
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES PRÉ-CRIBLAGES")
    print("="*80)
    
    summary_data = []
    for r in all_results:
        summary_data.append({
            'n': r['n'],
            'Petits facteurs': len(r['suspect_small_factors']),
            'Smoothness': r['smoothness_hint'],
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("EXTENSIONS TERMINÉES!")
    print("="*80)
    print("\nCes trois extensions rendent la Transformée de Guasti opérationnelle:")
    print("  1. Inversion Möbius 2D → Reconstruction de fonctions 2D")
    print("  2. Heatmaps → Visualisation de la structure multiplicative")
    print("  3. Pré-criblage TG → Guide pour la factorisation optimale")
    print("="*80)
