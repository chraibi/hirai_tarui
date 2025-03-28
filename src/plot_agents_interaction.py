import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
# c1
cn0 = -0.5
cr0 = 1.0
beta = 0.5
nu = 1.0
gamma = 2.0
epsilon = 3.0

# h1
hr0 = 1.0
lam = 1.5
sigma = 2.5

# c2, h2
phi1 = np.pi / 6
phi2 = np.pi / 3
phi3 = 2 * np.pi / 3
phi4 = 5 * np.pi / 6
cphi1 = hphi1 = 1.0
cphi2 = hphi2 = 0.5


# --- Functions ---
def c1(r):
    if r < beta:
        return cn0 + (0 - cn0) * (r / beta)
    elif r < nu:
        return cr0 * (r - beta) / (nu - beta)
    elif r < gamma:
        return cr0
    elif r < epsilon:
        return cr0 * (1 - (r - gamma) / (epsilon - gamma))
    else:
        return 0.0


def h1(r):
    if r < lam:
        return hr0
    elif r < sigma:
        return hr0 * (1 - (r - lam) / (sigma - lam))
    else:
        return 0.0


def c2(phi):
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


def h2(phi):
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


# Generate data
r_vals = np.linspace(0, 4, 500)
phi_vals = np.linspace(0, np.pi, 500)
c1_vals = [c1(r) for r in r_vals]
h1_vals = [h1(r) for r in r_vals]
c2_vals = [c2(p) for p in phi_vals]
h2_vals = [h2(p) for p in phi_vals]

# Plotting
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14

fig, axs = plt.subplots(2, 2, figsize=(16, 8))

# c1(r_ij)
axs[0, 0].plot(r_vals, c1_vals, color="black", linewidth=2)
axs[0, 0].axhline(0, color="gray", linestyle="--", linewidth=1)
axs[0, 0].axhline(cr0, color="black", linestyle="dashed", alpha=0.3)
axs[0, 0].axhline(cn0, color="black", linestyle="dashed", alpha=0.3)
axs[0, 0].axvline(beta, color="black", linestyle="dashed", alpha=0.3)
axs[0, 0].axvline(nu, color="black", linestyle="dashed", alpha=0.3)
axs[0, 0].axvline(gamma, color="black", linestyle="dashed", alpha=0.3)
axs[0, 0].axvline(epsilon, color="black", linestyle="dashed", alpha=0.3)
axs[0, 0].text(beta, -0.9, r"$\beta$", ha="center")
axs[0, 0].text(nu, -0.9, r"$\nu$", ha="center")
axs[0, 0].text(gamma, -0.9, r"$\gamma$", ha="center")
axs[0, 0].text(epsilon, -0.9, r"$\varepsilon$", ha="center")
# axs[0, 0].text(-0.2, cn0, r"$c_{n0}$", va="center", ha="right")
# axs[0, 0].text(-0.2, cr0, r"$c_{r0}$", va="center", ha="right")
axs[0, 0].set_ylim(cn0 - 0.2, cr0 + 0.5)
axs[0, 0].set_xlim(0, 4)
axs[0, 0].set_title(r"$c_1(r_{ij})$")
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([cn0, 0, cr0])
axs[0, 0].set_yticklabels([r"$c_{n0}$", "0", r"$c_{r0}$"])

# h1(r_ij)
axs[1, 0].plot(r_vals, h1_vals, color="black", linewidth=2)
axs[1, 0].axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.3)
axs[1, 0].axhline(hr0, color="black", linestyle="dashed", alpha=0.3)
axs[1, 0].axvline(lam, color="black", linestyle="dashed", alpha=0.3)
axs[1, 0].axvline(sigma, color="black", linestyle="dashed", alpha=0.3)
axs[1, 0].text(lam, -0.3, r"$\lambda$", ha="center")
axs[1, 0].text(sigma, -0.3, r"$\sigma$", ha="center")
# axs[1, 0].text(0, hr0, r"$h_{r0}$", va="center", ha="right")
axs[1, 0].set_ylim(-0.2, hr0 + 0.5)
axs[1, 0].set_xlim(0, 4)
axs[1, 0].set_title(r"$h_1(r_{ij})$")
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([0, hr0])
axs[1, 0].set_yticklabels(["0", r"$h_{r0}$"])

# c2(phi_ij)
axs[0, 1].plot(phi_vals, c2_vals, color="black", linewidth=2)
axs[0, 1].axvline(phi1, color="black", linestyle="dashed", alpha=0.3)
axs[0, 1].axvline(phi2, color="black", linestyle="dashed", alpha=0.3)
axs[0, 1].axvline(phi3, color="black", linestyle="dashed", alpha=0.3)
axs[0, 1].axvline(phi4, color="black", linestyle="dashed", alpha=0.3)
axs[0, 1].axhline(cphi1, color="black", linestyle="dashed", alpha=0.3)
axs[0, 1].axhline(cphi2, color="black", linestyle="dashed", alpha=0.3)
axs[0, 1].text(phi1, -0.1, r"$\phi_1$", ha="center")
axs[0, 1].text(phi2, -0.1, r"$\phi_2$", ha="center")
axs[0, 1].text(phi3, -0.1, r"$\phi_3$", ha="center")
axs[0, 1].text(phi4, -0.1, r"$\phi_4$", ha="center")
# axs[0, 1].text(0, cphi1, r"$c_{\phi1}$", va="center", ha="right")
# axs[0, 1].text(0, cphi2, r"$c_{\phi2}$", va="center", ha="right")
axs[0, 1].set_xlim(0, np.pi)
axs[0, 1].set_ylim(0, 1.1)
axs[0, 1].set_title(r"$c_2(\phi_{ij})$")
axs[0, 1].set_xticks([0, np.pi / 2, np.pi])
axs[0, 1].set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])
axs[0, 1].set_yticks([0, cphi2, cphi1])
axs[0, 1].set_yticklabels(["0", r"$c_{\phi2}$", r"$c_{\phi1}$"])

# h2(phi_ij)
axs[1, 1].plot(phi_vals, h2_vals, color="black", linewidth=2)
axs[1, 1].axvline(phi1, color="black", linestyle="dashed", alpha=0.3)
axs[1, 1].axvline(phi2, color="black", linestyle="dashed", alpha=0.3)
axs[1, 1].axvline(phi3, color="black", linestyle="dashed", alpha=0.3)
axs[1, 1].axvline(phi4, color="black", linestyle="dashed", alpha=0.3)
axs[1, 1].axhline(hphi1, color="black", linestyle="dashed", alpha=0.3)
axs[1, 1].axhline(hphi2, color="black", linestyle="dashed", alpha=0.3)
axs[1, 1].text(phi1, -0.1, r"$\phi_1$", ha="center")
axs[1, 1].text(phi2, -0.1, r"$\phi_2$", ha="center")
axs[1, 1].text(phi3, -0.1, r"$\phi_3$", ha="center")
axs[1, 1].text(phi4, -0.1, r"$\phi_4$", ha="center")
# axs[1, 1].text(0, hphi1, r"$h_{\phi1}$", va="center", ha="right")
# axs[1, 1].text(0, hphi2, r"$h_{\phi2}$", va="center", ha="right")
axs[1, 1].set_xlim(0, np.pi)
axs[1, 1].set_ylim(0, 1.1)
axs[1, 1].set_title(r"$h_2(\phi_{ij})$")
axs[1, 1].set_xticks([0, np.pi / 2, np.pi])
axs[1, 1].set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])
axs[1, 1].set_yticks([0, hphi2, hphi1])
axs[1, 1].set_yticklabels(["0", r"$h_{\phi2}$", r"$h_{\phi1}$"])

fig.tight_layout()
plt.savefig("agents_interaction.png")
