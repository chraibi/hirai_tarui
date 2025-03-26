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
    signs: List[np.ndarray],
    eta: float = 1.0,
    vision_radius: float = 1.5,
    fov_angle: float = np.pi * 2 / 3,  # 120 degrees
) -> np.ndarray:
    """Equation (9): Influence of visible signs within field of view and radius.

    This is immediate memory-based attraction.
    """
    force = np.zeros(2)
    for P_k in signs:
        dir_vec = P_k - x_i
        dist = np.linalg.norm(dir_vec)
        if dist <= vision_radius:
            angle = angle_between(v_i, dir_vec)
            if angle <= fov_angle / 2:
                force += eta * dir_vec / dist
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
        return q2 * random_unit()
    else:
        return q1 * random_unit()
