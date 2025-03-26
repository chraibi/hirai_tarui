"""Agent Class"""

import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from .parameters import ForceParameters

from .forces import (
    F_ai,
    F_bi,
    F_ci,
    F_wi,
    F_eik,
    F_fik,
    F_gi,
    F_hi,
    F_31,
    c_func,
    h_func,
)


class Agent:
    def __init__(
        self,
        position: List[float],
        velocity: List[float],
        mass: float = 1.0,
        damping: float = 0.5,
        params=None,
    ):
        """Create a new agent.

        Args:
            position: Initial position as [x, y].
            velocity: Initial velocity as [vx, vy].
            mass: Agent mass.
            damping: Viscous damping coefficient.
        """
        self.x = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)
        self.m = mass
        self.nu = damping
        self.acc = np.zeros(2)
        self.params = params or ForceParameters()

    def update(self, dt: float):
        """Update position and velocity using current acceleration."""
        self.v += dt * self.acc
        self.x += dt * self.v

    def compute_forces(
        self,
        others: List[Tuple[np.ndarray, np.ndarray]],
        polygons: List[Polygon],
        signs: List[np.ndarray],
        mem_signs: List[np.ndarray],
        exits: List[Polygon],
        h_i: np.ndarray,
    ):
        """Compute total force acting on the agent using the model equations.

        Equations used: (2) to (11) from Hirai and Tarui's model.

        Args:
            others: List of other agents as (position, velocity).
            polygons: List of polygons representing walls and obstacles.
            signs: Positions of visible signs.
            mem_signs: Positions of memorized signs.
            exits: Exit areas as polygons.
            h_i: External influence (herding).
        """
        f_ai = F_ai(self.v, a=self.params.a)
        f_bi = F_bi(self.x, self.v, others, c_func)
        f_ci = F_ci(self.x, self.v, others, h_func)

        f_wi = F_wi(
            self.x,
            self.v,
            polygons,
            d=self.params.wall_distance,
            w0=self.params.wall_strength_into,
            w1=self.params.wall_strength_always,
        )

        f_eik = F_eik(self.x, signs, eta=self.params.eta_sign)
        f_fik = F_fik(self.x, mem_signs, eta=self.params.eta_mem)
        f_gi = F_gi(self.x, exits, strength=self.params.exit_strength)
        f_hi = F_hi(h_i)
        di = (
            min([wall.exterior.distance(Point(self.x)) for wall in polygons])
            if polygons
            else 1.0
        )
        bwi = np.dot(self.v, self.v)
        f_31 = F_31(
            di,
            bwi,
            q1=self.params.q1,
            q2=self.params.q2,
            d=self.params.random_threshold,
        )

        F11 = f_ai + f_bi + f_ci
        F21 = f_wi + f_eik + f_fik + f_gi + f_hi
        F_total = F11 + F21 + f_31
        # debug all forces
        debug = 0
        if debug:
            print("f_ai", f_ai)
            print("f_bi", f_bi)
            print("f_ci", f_ci)
            print("f_wi", f_wi)
            print("f_eik", f_eik)
            print("f_fik", f_fik)
            print("f_gi", f_gi)
            print("f_hi", f_hi)
            print("f_31", f_31)
            print("F11", F11)
            print("F21", F21)
            print("F_total", F_total)

        self.acc = (F_total - self.nu * self.v) / self.m
