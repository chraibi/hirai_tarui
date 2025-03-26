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
