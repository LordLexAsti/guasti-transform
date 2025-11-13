# The Guasti Transform (La Transformée de Guasti)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TriadIA Protocol](https://img.shields.io/badge/Validated%20by-TriadIA-purple)](https://github.com/LordLexAsti)

**A Geometric Framework for Multiplicative Number Theory**

> "When multiplication tables reveal the hidden geometry of prime numbers."

## 🌟 Overview

The **Guasti Transform** is a novel mathematical framework that maps integers to geometric signatures based on their divisibility properties. Developed over **15 years of independent research** by Alexandre Guasti and formalized in **one day** using the TriadIA (Human-AI) collaboration protocol, this project offers a new perspective on the distribution of prime numbers and the Riemann Hypothesis.

## 📐 Key Concepts

### 1. The Guasti Grid
A 2D discrete space where a point $(i, j)$ is active if and only if $i$ divides $j$. This creates a "multiplicative landscape" rather than a simple number line.

### 2. The Law of Cotangent
We postulate that every divisor $d$ of a number $n$ corresponds to a ray with a specific angle $\theta$:
$$\theta = \arctan(1/d)$$

### 3. The Angular Signature
Every integer $n$ possesses a unique "fingerprint" called its Angular Signature ($S_A$):
$$S_A(n) = \{ \arctan(1/d) \mid d \in \text{Divisors}(n) \}$$
* **Prime Numbers** have a minimal signature (Low Entropy).
* **Composite Numbers** have complex signatures (High Entropy).

### 4. Riemann Zero Detection
By analyzing the spectral density of these angular signatures across the grid, we have experimentally detected resonance frequencies corresponding to the first zeros of the Riemann Zeta function.

## 🚀 Installation

```bash
git clone [https://github.com/LordLexAsti/guasti-transform.git](https://github.com/LordLexAsti/guasti-transform.git)
cd guasti-transform
pip install -r requirements.txt