"""Interaction Functions"""

import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Tuple

from .utils import normalize, angle_between, extract_segments, random_unit


def c_func(dist: float, angle: float) -> float:
    """Function c(dist, angle) from the paper. Equation (4)."""
    return np.exp(-dist) * np.cos(angle)


def h_func(dist: float, angle: float) -> float:
    """Function h(dist, angle) from the paper. Equation (5)."""
    return np.exp(-dist) * np.cos(angle)


# --- Force Components ---
def F_ai(velocity: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Equation (2): Individual's own driving force."""
    return a * normalize(velocity)


def F_bi(
    x_i: np.ndarray,
    v_i: np.ndarray,
    others: List[Tuple[np.ndarray, np.ndarray]],
    c_func,
) -> np.ndarray:
    """Equation (4): Influence from surrounding individuals."""
    force = np.zeros(2)
    for x_j, v_j in others:
        r_ij = x_j - x_i
        dist = np.linalg.norm(r_ij)
        angle = angle_between(v_i, r_ij)
        if dist > 0:
            c = c_func(dist, angle)
            force += -c * (r_ij / dist)
    return force


def F_ci(
    x_i: np.ndarray,
    v_i: np.ndarray,
    others: List[Tuple[np.ndarray, np.ndarray]],
    h_func,
) -> np.ndarray:
    """Equation (5): Cohesion with group members."""
    force = np.zeros(2)
    M = len(others)
    if M == 0:
        return force
    for x_j, v_j in others:
        r_ij = x_j - x_i
        dist = np.linalg.norm(r_ij)
        angle = angle_between(v_i, r_ij)
        if dist > 0:
            h = h_func(dist, angle)
            force += h * (r_ij / dist)
    return -force / M


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
        direction = normalize(x_i - np.array(closest_point.coords[0]))
        v_perp = np.dot(v_i, direction)
        strength = w1 if v_perp > 0 else w0
        return strength * direction * (1 - min_dist / d)

    return np.zeros(2)


def F_eik(x_i: np.ndarray, signs: List[np.ndarray], eta: float = 1.0) -> np.ndarray:
    """Equation (7): Influence of visible signs."""
    force = np.zeros(2)
    for P_k in signs:
        dir_vec = P_k - x_i
        dist = np.linalg.norm(dir_vec)
        if dist > 0:
            force += eta * dir_vec / dist
    return force


def F_fik(x_i: np.ndarray, mem_signs: List[np.ndarray], eta: float = 1.0) -> np.ndarray:
    """Equation (8): Influence of memorized signs."""
    return F_eik(x_i, mem_signs, eta)


def F_gi(
    x_i: np.ndarray, exit_polygons: List[Polygon], strength: float = 0.5
) -> np.ndarray:
    """Equation (9): Attraction toward exit polygon centroid."""
    if not exit_polygons:
        return np.zeros(2)
    center = np.array(exit_polygons[0].centroid.coords[0])
    return strength * normalize(center - x_i)


def F_hi(h_i: np.ndarray) -> np.ndarray:
    """Equation (10): Herding force (external influence)."""
    return h_i


def F_31(
    di: float, bwi: float, q1: float = 1.0, q2: float = 2.0, d: float = 1.0
) -> np.ndarray:
    """Equation (11): Random fluctuation force."""
    if di > d:
        return q1 * random_unit()
    elif bwi > 0:
        return -q2 * random_unit()
    else:
        return -q1 * random_unit()
