import numpy as np
from shapely.geometry import Polygon, Point

# --- Utility Functions ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)

# --- Interaction Functions (Distance and Angle Dependent) ---
def c_func(dist, angle):
    return np.exp(-dist) * np.cos(angle)

def h_func(dist, angle):
    return np.exp(-dist) * np.cos(angle)

# --- Force Components ---
def F_ai(velocity, a=1.0):
    return a * normalize(velocity)

def F_bi(x_i, v_i, others, c_func):
    force = np.zeros(2)
    for x_j, v_j in others:
        r_ij = x_j - x_i
        dist = np.linalg.norm(r_ij)
        angle = angle_between(v_i, r_ij)
        if dist > 0:
            c = c_func(dist, angle)
            force += -c * (r_ij / dist)
    return force

def F_ci(x_i, v_i, others, h_func):
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

def F_wi(x_i, v_i, walls, d=1.0, w0=1.0, w1=2.0):
    for wall in walls:
        point = Point(x_i)
        dist = wall.exterior.distance(point)
        if dist < d:
            direction = np.array(wall.exterior.centroid.coords[0]) - x_i
            direction = normalize(-direction)
            v_perp = np.dot(v_i, direction)
            if v_perp > 0:
                return w1 * direction
            else:
                return w0 * direction
    return np.zeros(2)

def F_eik(x_i, signs, eta=1.0):
    force = np.zeros(2)
    for P_k in signs:
        dir_vec = P_k - x_i
        dist = np.linalg.norm(dir_vec)
        if dist > 0:
            force += eta * dir_vec / dist
    return force

def F_fik(x_i, memorized_signs, eta=1.0):
    return F_eik(x_i, memorized_signs, eta)

def F_gi(g_i):
    return g_i

def F_hi(h_i):
    return h_i

def F_31(di, bwi, q1=1.0, q2=2.0, d=1.0):
    if di > d:
        return q1 * random_unit()
    elif bwi > 0:
        return -q2 * random_unit()
    else:
        return -q1 * random_unit()

def angle_between(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    cos_theta = np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0)
    return np.arccos(cos_theta)

def random_unit():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])

# --- Agent Class ---
class Agent:
    def __init__(self, position, velocity, mass=1.0, damping=0.5):
        self.x = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)
        self.m = mass
        self.nu = damping
        self.acc = np.zeros(2)

    def update(self, dt):
        self.v += dt * self.acc
        self.x += dt * self.v

    def compute_forces(self, others, walls, signs, mem_signs, g_i, h_i):
        f_ai = F_ai(self.v)
        f_bi = F_bi(self.x, self.v, others, c_func)
        f_ci = F_ci(self.x, self.v, others, h_func)
        f_wi = F_wi(self.x, self.v, walls)
        f_eik = F_eik(self.x, signs)
        f_fik = F_fik(self.x, mem_signs)
        f_gi = F_gi(g_i)
        f_hi = F_hi(h_i)
        di = min([wall.exterior.distance(Point(self.x)) for wall in walls]) if walls else 1.0
        bwi = np.dot(self.v, self.v)  # Simplified proxy
        f_31 = F_31(di, bwi)

        F11 = f_ai + f_bi + f_ci
        F21 = f_wi + f_eik + f_fik + f_gi + f_hi
        F_total = F11 + F21 + f_31

        self.acc = (F_total - self.nu * self.v) / self.m

# --- Simulation Setup ---
def simulate():
    dt = 0.1
    steps = 100
    agent = Agent(position=[0.0, 0.0], velocity=[1.0, 0.0])

    # One other agent for interaction test
    other_agent = Agent(position=[1.0, 1.0], velocity=[0.0, -1.0])

    # Walls
    walls = [Polygon([(5,5), (5,-5), (6,-5), (6,5)])]

    # Signs
    signs = [np.array([10.0, 0.0])]
    mem_signs = []

    # Constant exit and panic force
    g_i = np.array([0.0, 0.0])
    h_i = np.array([0.0, 0.0])

    traj = []
    for _ in range(steps):
        agent.compute_forces(others=[(other_agent.x, other_agent.v)], walls=walls,
                             signs=signs, mem_signs=mem_signs,
                             g_i=g_i, h_i=h_i)
        agent.update(dt)
        traj.append(agent.x.copy())

    return np.array(traj)

if __name__ == "__main__":
    traj = simulate()
    import matplotlib.pyplot as plt
    plt.plot(traj[:,0], traj[:,1], label="Agent Path")
    plt.legend()
    plt.axis("equal")
    plt.title("Hirai-Tarui Model Simulation (1 Agent)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
