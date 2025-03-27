import numpy as np
import matplotlib.pyplot as plt


# --- c1(r_ij): distance-based repulsion
def c1(r, cn0=-0.5, cr0=1.0, beta=1.0, gamma=2.0, epsilon=3.0):
    if r < beta:
        return cn0 + (cr0 - cn0) * (r / beta)
    elif r < gamma:
        return cr0
    elif r < epsilon:
        return cr0 * (1 - (r - gamma) / (epsilon - gamma))
    else:
        return 0.0


# --- h1(r_ij): distance-based cohesion
def h1(r, hr0=1.0, lam=2.0, sigma=3.0):
    if r < lam:
        return hr0
    elif r < sigma:
        return hr0 * (1 - (r - lam) / (sigma - lam))
    else:
        return 0.0


# --- c2(phi_ij): angle-based repulsion
def c2(
    phi,
    cphi1=1.0,
    cphi2=0.5,
    phi1=np.pi / 6,
    phi2=np.pi / 3,
    phi3=2 * np.pi / 3,
    phi4=5 * np.pi / 6,
):
    if phi < phi1:
        return cphi1
    elif phi < phi2:
        return cphi1 - (cphi1 - cphi2) * (phi - phi1) / (phi2 - phi1)
    elif phi < phi3:
        return cphi2
    elif phi < phi4:
        return cphi2 * (1 - (phi - phi3) / (phi4 - phi3))
    else:
        return 0.0


# --- h2(phi_ij): angle-based cohesion (same structure as c2)
def h2(
    phi,
    hphi1=1.0,
    hphi2=0.5,
    phi1=np.pi / 6,
    phi2=np.pi / 3,
    phi3=2 * np.pi / 3,
    phi4=5 * np.pi / 6,
):
    if phi < phi1:
        return hphi1
    elif phi < phi2:
        return hphi1 - (hphi1 - hphi2) * (phi - phi1) / (phi2 - phi1)
    elif phi < phi3:
        return hphi2
    elif phi < phi4:
        return hphi2 * (1 - (phi - phi3) / (phi4 - phi3))
    else:
        return 0.0


# Generate data to plot
r_values = np.linspace(0, 4, 200)
phi_values = np.linspace(0, np.pi, 200)

c1_vals = [c1(r) for r in r_values]
h1_vals = [h1(r) for r in r_values]
c2_vals = [c2(phi) for phi in phi_values]
h2_vals = [h2(phi) for phi in phi_values]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].plot(r_values, c1_vals, label="$c_1(r_{ij})$")
axs[0, 0].set_title("$c_1(r_{ij})$")
axs[0, 0].set_xlabel("$r_{ij}$")
axs[0, 0].set_ylabel("Strength")
axs[0, 0].axhline(0, color="gray", linestyle="--", linewidth=1)  # y=0 line

axs[0, 1].plot(phi_values, c2_vals, label="$c_2(\\phi_{ij})$")
axs[0, 1].set_title("$c_2(\\phi_{ij})$")
axs[0, 1].set_xlabel("$\\phi_{ij}$")
axs[0, 1].set_ylabel("Strength")

axs[1, 0].plot(r_values, h1_vals, label="$h_1(r_{ij})$")
axs[1, 0].set_title("$h_1(r_{ij})$")
axs[1, 0].set_xlabel("$r_{ij}$")
axs[1, 0].set_ylabel("Strength")

axs[1, 1].plot(phi_values, h2_vals, label="$h_2(\\phi_{ij})$")
axs[1, 1].set_title("$h_2(\\phi_{ij})$")
axs[1, 1].set_xlabel("$\\phi_{ij}$")
axs[1, 1].set_ylabel("Strength")

fig.tight_layout()
plt.show()
