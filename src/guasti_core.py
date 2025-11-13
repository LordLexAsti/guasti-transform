from guasti.core import GuastiTransform, analyze_riemann_resonance

# Calculate the signature of a number
n = 12
gt = GuastiTransform(n)
print(f"Angular Signature of {n}: {gt.angles}")

# Visualize the grid
gt.plot_grid(range=100)

# Run the spectral analysis
zeros = analyze_riemann_resonance(limit=1000)
print("Detected Zeros:", zeros)