import numpy as np
import matplotlib.pyplot as plt

def count_positive_real_roots(alpha, rho, beta=1.0):
    # Equation: β²y³ - αβ²y² + ρy - α = 0
    coeffs = [beta**2, -alpha * beta**2, rho, -alpha]
    roots = np.roots(coeffs)
    # Count positive, real roots
    return sum(r.real > 0 and np.isreal(r) for r in roots)

# Define parameter ranges
alpha_vals = np.linspace(4.5, 10, 1000)
rho_vals = np.linspace(5, 20, 1000)

# Prepare grid
Z = np.zeros((len(alpha_vals), len(rho_vals)))

# Fill grid with root counts
for i, alpha in enumerate(alpha_vals):
    for j, rho in enumerate(rho_vals):
        Z[i, j] = count_positive_real_roots(alpha, rho)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(rho_vals, alpha_vals, Z, levels=[0.5, 1.5, 2.5, 3.5],
             colors=['white', 'white', '#888888'], alpha=0.8)

# Labels for the regions
plt.text(8, 7, rf'$Monostable$', fontsize=16, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white'))
plt.text(18, 8.5, rf'$Bistable$', fontsize=16, color="white", ha='center', va='center')

# plt.colorbar(ticks=[1, 3], label='Number of Positive Real Roots')
plt.xlabel(r'$\rho$', fontsize=16)
plt.ylabel(r'$\alpha$', fontsize=16)
plt.tick_params(labelsize=14)  # tick font size
# plt.title('Phase Diagram of GFP Steady States (TMG Induction)')
# plt.grid(True)
plt.tight_layout()
plt.show()
