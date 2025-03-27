"""Interaction Functions"""

import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Tuple

from .utils import normalize, angle_between, extract_segments, random_unit


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


def F_ci(
    x_i: np.ndarray,
    v_i: np.ndarray,
    others: List[Tuple[np.ndarray, np.ndarray]],
    h1_func,
    h2_func,
) -> np.ndarray:
    """
    Equation (5): Cohesion force based on velocity alignment.

    F_ci = (1/M) * sum_j [ h(x_i, v_i, x_j) * (v_j - v_i) ]

    where h = h1(r_ij) * h2(phi_ij)

    Args:
        x_i: Position of agent i
        v_i: Velocity of agent i
        others: List of (x_j, v_j)
        h1_func: function of distance
        h2_func: function of angle

    Returns:
        np.ndarray: Cohesion force vector
    """
    force = np.zeros(2)
    M = len(others)
    if M == 0:
        return force

    for x_j, v_j in others:
        r_ij = x_j - x_i
        dist = np.linalg.norm(r_ij)
        if dist > 0:
            phi_ij = angle_between(v_i, r_ij)
            h = h1_func(dist) * h2_func(phi_ij)
            force += h * (v_j - v_i)

    return force / M


def F_bi(
    x_i: np.ndarray,
    v_i: np.ndarray,
    others: List[Tuple[np.ndarray, np.ndarray]],
    c1_func,
    c2_func,
) -> np.ndarray:
    """
    Equation (4): Repulsion from surrounding individuals.

    F_bi = sum_j [ c(x_i, v_i, x_j) * (x_j - x_i) / ||x_j - x_i|| ]

    where c = c1(r_ij) * c2(phi_ij)

    Args:
        x_i: Position of agent i
        v_i: Velocity of agent i
        others: List of (x_j, v_j)
        c1_func: function of distance
        c2_func: function of angle

    Returns:
        np.ndarray: Repulsion force vector
    """
    force = np.zeros(2)
    for x_j, _ in others:
        r_ij = x_j - x_i
        dist = np.linalg.norm(r_ij)
        if dist > 0:
            phi_ij = angle_between(v_i, r_ij)
            c = c1_func(dist) * c2_func(phi_ij)
            force += c * (r_ij / dist)
    return -force


# --- Force Components ---
def F_ai(velocity: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Equation (2): Individual's own driving force."""
    return a * normalize(velocity)


def F_wi(
    x_i: np.ndarray,
    v_i: np.ndarray,
    polygons: List[Polygon],
    d: float = 1.0,
    w0: float = 6.0,
    w1: float = 6.0,
) -> np.ndarray:
    """Equation (6): Repulsive force from nearby walls or obstacles."""
    point = Point(x_i)
    min_dist = float("inf")
    closest_wall = None
    wall_segments = [
        segment for polygon in polygons for segment in extract_segments(polygon)
    ]

    for segment in wall_segments:
        dist = segment.distance(point)
        if dist < min_dist:
            min_dist = dist
            closest_wall = segment

    if closest_wall is not None and min_dist < d:
        closest_point = closest_wall.interpolate(closest_wall.project(point))
        e_w = normalize(x_i - np.array(closest_point.coords[0]))  # away from the wall
        v_wi = -np.dot(v_i, e_w)  # sign convention from paper: into wall = positive
        if v_wi > 0:
            strength = (w0 * v_wi * (d - min_dist) / d) + w1
        else:
            strength = w1

        return strength * e_w, e_w

    return np.zeros(2), np.array([1.0, 0.0])  # arbitrary unit vector (not used)


def F_eik(
    x_i: np.ndarray,
    v_i: np.ndarray,
    signs: List[Tuple[np.ndarray, np.ndarray]],
    eta: float = 1.0,
    vision_radius: float = 1.5,
    fov_angle: float = np.pi * 2 / 3,  # agent's field of view
    sign_fov: float = np.pi * 0.5,  # sign's "facing cone"
) -> np.ndarray:
    """Equation (9): Influence of visible signs with directional constraints.

    A sign attracts an agent only if:
    - the agent is within the sign's vision radius
    - the agent is within the sign's directional cone
    - the sign is within the agent's field of view
    """
    force = np.zeros(2)
    for sign_position, sign_direction in signs:
        to_agent = x_i - sign_position
        to_sign = sign_position - x_i

        dist = np.linalg.norm(to_agent)
        if dist > vision_radius:
            continue

        angle_agent_to_sign = angle_between(sign_direction, to_agent)
        angle_sign_to_agent = angle_between(v_i, to_sign)

        if angle_agent_to_sign <= sign_fov / 2 and angle_sign_to_agent <= fov_angle / 2:
            force += eta * normalize(sign_position - x_i)

    return force


def F_fik(x_i: np.ndarray, mem_signs: List[np.ndarray], eta: float = 1.0) -> np.ndarray:
    """Equation (9) modified: Influence of memorized signs.

    Unlike F_eik this is a persistent memory-based attraction.
    """
    force = np.zeros(2)
    for P_k in mem_signs:
        dir_vec = P_k - x_i
        dist = np.linalg.norm(dir_vec)
        if dist > 0:
            force += eta * dir_vec / dist
    return force


def F_gi(
    x_i: np.ndarray, exit_polygons: List[Polygon], strength: float = 0.5
) -> np.ndarray:
    """Equation (9): Attraction toward exit polygon centroid."""
    if not exit_polygons:
        return np.zeros(2)
    center = np.array(exit_polygons[0].centroid.coords[0])
    return strength * normalize(center - x_i)


def F_hi(
    x_i: np.ndarray, x_panic: np.ndarray, strength: float = 1.0, cuttof: float = 20
) -> np.ndarray:
    """
    Equation (10): Herding or panic repulsion force.
    Applies a constant force of magnitude `strength` away from the panic site.

    Parameters:
    - x_i: Position of individual i (2D or 3D vector).
    - x_panic: Position of panic center (2D or 3D vector).
    - strength: Magnitude of the force.

    Returns:
    - A force vector pointing from the panic site to the individual, with magnitude `strength`.
    """
    direction = x_i - x_panic
    distance = np.linalg.norm(direction)
    if distance == 0 or distance > cuttof:
        return np.zeros_like(x_i)  # no direction if exactly at panic site

    return strength * (direction / distance)


def F_31(
    di: float, bwi: float, q1: float = 1.0, q2: float = 2.0, d: float = 1.0
) -> np.ndarray:
    """Equation (11): Random fluctuation force."""
    if di > d:
        return q1 * random_unit()
    elif bwi > 0:
        return q2 * random_unit()
    else:
        return q1 * random_unit()
