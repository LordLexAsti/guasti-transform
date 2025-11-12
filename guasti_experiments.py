# Retry with simplified (non-mathtext) titles to avoid parser issues.

import math
import cmath
from math import log, sqrt, atan2, isfinite, pi, floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sieve_divisor_counts(N: int):
    d = [0]*(N+1)
    for i in range(1, N+1):
        step = i
        for j in range(i, N+1, step):
            d[j] += 1
    return d

def sieve_mobius(N: int):
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

def dirichlet_divisor_prefix(N: int):
    d = sieve_divisor_counts(N)
    D = np.cumsum(d[1:])
    x = np.arange(1, N+1, dtype=float)
    gamma = 0.577215664901532860606512090082402431
    Delta = D - (x*np.log(x) + (2*gamma - 1.0)*x)
    return x, D, Delta

def plot_experiment_1(N: int = 20000):
    x, D, Delta = dirichlet_divisor_prefix(N)
    plt.figure(figsize=(10,6))
    plt.plot(x, Delta, linewidth=0.8)
    plt.title("Experiment 1: Delta(x) for Dirichlet Divisor Sum", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Delta(x) = D(x) - x*log(x) - (2*gamma-1)*x", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/experiment1_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    df = pd.DataFrame({"x": x[::max(1, N//10)], "D(x)": D[::max(1, N//10)], "Delta(x)": Delta[::max(1, N//10)]})
    return df

def dirichlet_characters_mod_q(q: int):
    if q == 3:
        def chi0(n):
            return 0 if math.gcd(n,3)>1 else 1
        def chi1(n):
            if math.gcd(n,3)>1: return 0
            return 1 if n%3==1 else -1
        return [chi0, chi1]

    if q == 4:
        def chi0(n):
            return 0 if math.gcd(n,4)>1 else 1
        def chi1(n):
            if math.gcd(n,4)>1: return 0
            return 1 if ((n-1)//2) % 2 == 0 else -1
        return [chi0, chi1]

    if q == 5:
        g = 2
        exp_table = {1:0}
        cur = 1
        for e in range(1,4):
            cur = (cur*g) % 5
            exp_table[cur] = e
        def chi_k_factory(k):
            def chi(n):
                if math.gcd(n,5)>1:
                    return 0
                a = n % 5
                e = exp_table[a]
                return cmath.exp(2j*math.pi*k*e/4.0)
            return chi
        return [chi_k_factory(k) for k in range(4)]

    if q == 8:
        def principal(n):
            return 0 if math.gcd(n,8)>1 else 1
        def chiA(n):
            if math.gcd(n,8)>1: return 0
            return 1 if ((n-1)//2) % 2 == 0 else -1
        def chiB(n):
            if math.gcd(n,8)>1: return 0
            return 1 if (((n*n - 1)//8) % 2) == 0 else -1
        def chiAB(n):
            a = chiA(n)
            b = chiB(n)
            if a == 0 or b == 0: return 0
            return a*b
        return [principal, chiA, chiB, chiAB]

    raise ValueError("Only q in {3,4,5,8} supported.")

def S_chi_prefix(X: int, q: int, chi_index: int):
    chis = dirichlet_characters_mod_q(q)
    if chi_index < 0 or chi_index >= len(chis):
        raise ValueError("chi_index out of range for modulus q.")
    chi = chis[chi_index]
    chi_vals = [0]*(X+1)
    for n in range(1, X+1):
        chi_vals[n] = chi(n)
    S = np.zeros(X+1, dtype=complex)
    for i in range(1, X+1):
        ci = chi_vals[i]
        if ci == 0: 
            continue
        maxj = X//i
        for j in range(1, maxj+1):
            cj = chi_vals[j]
            if cj == 0: 
                continue
            t = i*j
            S[t] += ci * np.conjugate(cj)
    S_prefix = np.cumsum(S[1:])
    tgrid = np.arange(1, X+1)
    return tgrid, S_prefix

def plot_experiment_2(X: int = 20000, q: int = 5, chi_index: int = 1):
    t, S = S_chi_prefix(X, q, chi_index)
    plt.figure(figsize=(10,6))
    plt.plot(t, S.real, linewidth=0.8)
    plt.title(f"Experiment 2: S_chi(t) for modulus q={q}, chi_index={chi_index}", fontsize=14)
    plt.xlabel("t", fontsize=12)
    plt.ylabel("Re S_chi(t)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/experiment2_angular.png', dpi=150, bbox_inches='tight')
    plt.close()
    df = pd.DataFrame({"t": t[::max(1, X//10)], "Re S_chi(t)": S.real[::max(1, X//10)], "Im S_chi(t)": S.imag[::max(1, X//10)]})
    return df

def mobius_filtered_angular_sum(X: int, theta0: float, delta: float):
    mu = sieve_mobius(X)
    total = 0
    cnt = 0
    for i in range(1, X+1):
        mui = mu[i]
        if mui == 0: 
            continue
        maxj = X//i
        for j in range(1, maxj+1):
            muj = mu[j]
            if muj == 0:
                continue
            ang = math.atan2(j, i)
            if abs(ang - theta0) <= delta:
                total += mui*muj
                cnt += 1
    return total, cnt

def plot_experiment_3(X: int = 12000, bins: int = 48):
    thetas = np.linspace(0.0, math.pi/2, bins)
    half_width = (math.pi/2) / (2*bins)
    values = []
    counts = []
    for th in thetas:
        s, c = mobius_filtered_angular_sum(X, th, half_width)
        values.append(s)
        counts.append(c)
    thetas_deg = thetas * 180.0 / math.pi
    plt.figure(figsize=(10,6))
    plt.plot(thetas_deg, values, linewidth=0.8)
    plt.title(f"Experiment 3: Mobius-filtered sum vs angle (X={X})", fontsize=14)
    plt.xlabel("Angle (degrees)", fontsize=12)
    plt.ylabel("Sum of mu(i) mu(j) in cone", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/experiment3_mobius.png', dpi=150, bbox_inches='tight')
    plt.close()
    df = pd.DataFrame({
        "theta_deg": thetas_deg,
        "mobius_sum": values,
        "pair_count_in_cone": counts
    })
    return df

# Run the three experiments with defaults
print("Running Experiment 1: Hyperbolic counting (Delta function)...")
df1 = plot_experiment_1(N=20000)
print(df1.to_string())
print("\n" + "="*80 + "\n")

print("Running Experiment 2: Angular tomography with Dirichlet characters...")
df2 = plot_experiment_2(X=20000, q=5, chi_index=1)
print(df2.to_string())
print("\n" + "="*80 + "\n")

print("Running Experiment 3: Mobius-filtered angular sweep...")
df3 = plot_experiment_3(X=12000, bins=48)
print(df3.to_string())
print("\n" + "="*80 + "\n")

print("All experiments completed! Visualizations saved to /mnt/user-data/outputs/")
