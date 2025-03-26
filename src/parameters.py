from dataclasses import dataclass
@dataclass
class ForceParameters:
    # Driving force
    a: float = 1.0

    # Wall interaction
    wall_distance: float = 1.0
    wall_strength_close: float = 6.0
    wall_strength_far: float = 6.0

    # Sign attraction
    eta_sign: float = 1.0
    eta_mem: float = 1.0

    # Exit attraction
    exit_strength: float = 0.5

    # Random force
    q1: float = 1.0
    q2: float = 2.0
    random_threshold: float = 1.0
    hi: float = 1.0