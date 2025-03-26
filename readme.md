# Hirai and Tarui Crowd Simulation Model

This repository implements the pedestrian dynamics model proposed by **Hirai and Tarui** (1975) for simulating the behavior of crowds.

---

## Mathematical Model

The motion of each individual $i$ is governed by the following second-order differential equation:


$$F_{11} + F_{21} + F_{31}=m_i \ddot{x}_i+ \nu_i \dot{x}_i$$

Where:
- $x_i$ is the position vector
- $\dot{x}_i$ is the velocity
- $m_i$ is the mass
- $\nu_i$ is the viscosity (damping) coefficient
- $F_{11}$: social forces
- $F_{21}$: environmental forces
- $F_{31}$: random force due to disturbances

---

## Components of the Forces

### 1. **Social Force** $F_{11} = F_{ai} + F_{bi} + F_{ci}$

#### (a) Forward Driving Force:

$$F_{ai} = a \cdot \hat{x}_i$$

Pushes the individual forward in the direction of motion.

#### (b) Attraction / Repulsion:

$$F_{bi} = - \sum_j c(x_i, \dot{x}_i, x_j) \frac{x_j - x_i}{r^{ij}}$$


Attracts agents at a moderate distance, repels those that are too close.

$$c(x_i, \dot{x}_i, x_j) = c_1(r^{ij}) c_2(\theta^{ij})$$

#### (c) Velocity Matching (Alignment):

$$F_{ci} = - \frac{1}{M} \sum_j h(x_i, \dot{x}_i, x_j) \frac{x_j - x_i}{r^{ij}}$$

Encourages alignment of motion in local groups.

$$h(x_i, \dot{x}_i, x_j) = h_1(r^{ij}) h_2(\theta^{ij})$$

---

### 2. **Environmental Force** $F_{21}$

$$F_{21} = F_{wi} + \sum_k (F_{eik} + F_{fik}) + F_{gi} + F_{hi}$$

- **Wall Repulsion** $F_{wi}$:

Repels agents from walls:


$$
F_{wi} =
\begin{cases}
0 & \text{if } d_i > d \\\\
\left( w_0 v_{wi} \frac{(d - d_i)}{d} + w_1 \right) \cdot \mathbf{e}_w & \text{if } d_i < d \text{ and } v_{wi} > 0 \\\\
w_1 \cdot \mathbf{e}_w & \text{if } d_i < d \text{ and } v_{wi} \le 0
\end{cases}
$$

Where:

- $\mathbf{e}_w$ is the unit vector pointing **away from the wall**
- $v_{wi}$ is the component of velocity **along $-\mathbf{e}_w$** (i.e., *into* the wall is positive)
- $d_i$ is the **actual distance** from the agent to the wall
- $d$ is the **cutoff distance**: the maximum range in which the wall force acts

This force **slows down the agent as they approach the wall**, and **acts only within a threshold distance**.


![fig](figs/wall.png)


- **Attraction to Signs** $F_{eik}$:
  
$$F_{eik} = \eta \cdot  \frac{P_k - x_i}{|P_k - x_i|}$$

Attracts the agent toward visible guiding signs $P_k$.

- **Memory-Based Attraction to Signs** $F_{fik}$:
Same as $F_{eik}$, but persists after the sign is out of sight.

- **Exit Attraction** $F_{gi}$:

$$F_{gi} = g_i$$
Drives agents toward a known exit nearby.

### Summary of Region-Specific Forces

The model assumes that different forces act in mutually exclusive regions:

| Region                       | Active Forces                      |
|-----------------------------|------------------------------------|
| **Exit domain** (near exit) | `F_gi` (goal/exit attraction only) |
| **Visible sign domain**     | `F_eik` (only if sign is visible)  |
| **Memory domain**           | `F_fik` (only if sign was memorized) |
| **All other regions**       | Wall, social, cohesion, fluctuation forces |

> **Note**:  
`F_gi`, `F_eik`, and `F_fik` are **never active at the same time**.  
Each agent experiences only one of them depending on their location and memory state.



- **Panic Avoidance** $F_{hi}$:

$$F_{hi} = h_i$$

Pushes agents away from the panic origin.

---

### 3. **Random Force** $F_{31}$

Models stochastic disturbances:

$$F_{31} =
\begin{cases}
  q_1 \cdot \text{rand}() & \text{if } d_i > d \\
 -q_2 \cdot \text{rand}() & \text{if } d_i \leq d \text{ and } b_{wi} > 0 \\
 -q_1 \cdot \text{rand}() & \text{if } d_i \leq d \text{ and } b_{wi} \leq 0
\end{cases}
$$

Where `rand()` is a unit vector in a random direction.

---

## Implementation Notes
- Forces are modular Python functions for testing and reuse
- Agents interact with each other and with walls (modeled via Shapely)
- Basic Euler integration is used to update agent positions

---

## Future Extensions
- Calibration




https://github.com/user-attachments/assets/d58b6049-3dde-448e-a8e6-c35dedf4ad62


---

## References
- K. Hirai and K. Tarui, "A Simulation of the Behavior of a Crowd in Panic", Kobe University, 1975.
---

