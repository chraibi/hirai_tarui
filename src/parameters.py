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
    hi: float = 1.0

    vision_radius: float = 1.5
    fov_angle: float = np.pi * 2 / 3  # 120 degrees

    exit_domain_radius: float = 4.0
