"""Agent Class"""

import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from .parameters import ForceParameters
from .utils import angle_between
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
        self.mem_signs = []
        self.last_exit_seen = []
        self.params = params or ForceParameters()

    def update(self, dt: float):
        """Update position and velocity using current acceleration."""
        self.v += dt * self.acc
        self.x += dt * self.v

    def get_visible_signs(self, signs):
        visible_now = []
        for sign in signs:
            P_k = np.array(sign.centroid.coords[0])
            dist = np.linalg.norm(P_k - self.x)
            angle = angle_between(self.v, P_k - self.x)
            if (
                dist <= self.params.sign_vision_radius
                and angle <= self.params.fov_angle / 2
            ):
                visible_now.append(P_k)
        return visible_now

    def compute_forces(
        self,
        others: List[Tuple[np.ndarray, np.ndarray]],
        polygons: List[Polygon],
        signs: List[np.ndarray],
        exits: List[Polygon],
        x_panic: np.ndarray,
    ):
        """Compute total force acting on the agent using the model equations.

        Equations used: (2) to (11) from Hirai and Tarui's model.

        Args:
            others: List of other agents as (position, velocity).
            polygons: List of polygons representing walls and obstacles.
            signs: Positions of visible signs.
            exits: Exit areas as polygons.
            h_i: External influence (herding).
        """
        f_ai = F_ai(self.v, a=self.params.a)
        f_bi = F_bi(self.x, self.v, others, c_func)
        f_ci = F_ci(self.x, self.v, others, h_func)

        f_wi, e_w = F_wi(
            self.x,
            self.v,
            polygons,
            d=self.params.wall_distance,
            w0=self.params.wall_strength_into,
            w1=self.params.wall_strength_always,
        )
        ###################### signs and exits
        exit_centers = [np.array(exit.centroid.coords[0]) for exit in exits]
        min_exit_dist = min(np.linalg.norm(self.x - c) for c in exit_centers)
        self.last_exit_seen = np.argmin(
            [np.linalg.norm(self.x - c) for c in exit_centers]
        )

        if min_exit_dist <= self.params.exit_domain_radius:
            # Close to exit → apply only F_gi
            f_gi = F_gi(self.x, exits, strength=self.params.exit_strength)
            f_eik = np.zeros(2)
            f_fik = np.zeros(2)
        else:
            visible_now = self.get_visible_signs(signs)
            f_gi = np.zeros(2)
            # Memorize visible signs
            for P_k in visible_now:
                if not any(np.allclose(P_k, mem) for mem in self.mem_signs):
                    self.mem_signs.append(P_k)

            # Choose between visible-sign force and memorized-sign force (never both)
            if visible_now:
                f_eik = F_eik(
                    self.x,
                    self.v,
                    signs,
                    eta=self.params.eta_sign,
                    vision_radius=self.params.sign_vision_radius,
                    fov_angle=self.params.fov_angle,
                )
                f_fik = np.zeros(2)
            else:
                f_eik = np.zeros(2)
                f_fik = F_fik(self.x, self.mem_signs, eta=self.params.eta_mem)
        ##################### signs and exits

        f_hi = F_hi(self.x, x_panic, self.params.hi)

        di = (
            min([wall.exterior.distance(Point(self.x)) for wall in polygons])
            if polygons
            else 1.0
        )

        bwi = np.dot(f_wi, e_w)
        f_31 = F_31(
            di,
            bwi,
            q1=self.params.q1,
            q2=self.params.q2,
            d=self.params.wall_distance,
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
