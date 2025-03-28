import numpy as np
from dataclasses import dataclass


@dataclass
class ForceParameters:
    # Driving force
    a: float = 1.0

    # Wall interaction
    wall_distance: float = 1.0
    wall_strength_into: float = 6.0
    wall_strength_always: float = 6.0

    # Sign attraction
    eta_sign: float = 1.0
    eta_mem: float = 1.0

    # Exit attraction
    exit_strength: float = 0.5

    # Random force
    q1: float = 1.0
    q2: float = 2.0
    # panic force
    hi: float = 1.0
    cutoff_hi: float = 20
    sign_vision_radius: float = 1.5
    fov_angle: float = np.pi * 2 / 3  # 120 degrees

    exit_domain_radius: float = 4.0


@dataclass
class C1Parameters:
    cn0: float = -0.5
    cr0: float = 1.0
    beta: float = 0.5
    nu: float = 1.0
    gamma: float = 2.0
    epsilon: float = 3.0


@dataclass
class H1Parameters:
    hr0: float = 1.0
    lam: float = 1.5
    sigma: float = 2.5


@dataclass
class C2H2Parameters:
    phi1: float = np.pi / 6
    phi2: float = np.pi / 3
    phi3: float = 2 * np.pi / 3
    phi4: float = 5 * np.pi / 6
    cphi1: float = 1.0
    cphi2: float = 0.5
    hphi1: float = 1.0
    hphi2: float = 0.5
