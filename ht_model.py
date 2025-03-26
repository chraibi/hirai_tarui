import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry import LineString
from typing import List, Tuple


# --- Utility Functions ---
def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector. Returns a zero vector if norm is 0."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)


def extract_walls_from_geometry(geometry: Polygon) -> List[Polygon]:
    """Extract outer and hole boundaries of a polygon as individual polygons.
    Useful for converting a geometry into wall polygons.
    """
    wall_polys = [Polygon(geometry.exterior.coords)]
    wall_polys += [Polygon(interior.coords) for interior in geometry.interiors]
    return wall_polys


def extract_segments(polygon: Polygon) -> List[LineString]:
    """Extract all wall segments (edges) from a polygon."""
    coords = list(polygon.exterior.coords)
    return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    cos_theta = np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0)
    return np.arccos(cos_theta)


def random_unit() -> np.ndarray:
    """Return a random unit vector."""
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])


# --- Interaction Functions ---
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


# --- Agent Class ---
class Agent:
    def __init__(
        self,
        position: List[float],
        velocity: List[float],
        mass: float = 1.0,
        damping: float = 0.5,
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
        f_ai = F_ai(self.v)
        f_bi = F_bi(self.x, self.v, others, c_func)
        f_ci = F_ci(self.x, self.v, others, h_func)

        f_wi = F_wi(self.x, self.v, polygons)

        f_eik = F_eik(self.x, signs)
        f_fik = F_fik(self.x, mem_signs)
        f_gi = F_gi(self.x, exits)
        f_hi = F_hi(h_i)
        di = (
            min([wall.exterior.distance(Point(self.x)) for wall in polygons])
            if polygons
            else 1.0
        )
        bwi = np.dot(self.v, self.v)
        f_31 = F_31(di, bwi)

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
