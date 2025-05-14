# 3D Galaxy Simulation - Homework Project
# Created by Ashkan Ajrian for "Simulation and Modeling of Natural Processes" course.
# I developed this optimized 3D galaxy simulation using the clever Barnes-Hut algorithm. 
# While it was originally a course assignment, I've enhanced its performance significantly through several optimizations.


from numpy import array, zeros, sqrt, float64 as np_float64
from numpy.linalg import norm
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, float64, types

######### Numba-accelerated functions #######
@njit(float64[:](float64[:], float64[:], float64, float64))
def compute_force(pos1, pos2, m1, m2):
    cutoff = 0.002
    d = pos1 - pos2
    distance = sqrt(d[0]**2 + d[1]**2 + d[2]**2)
    if distance < cutoff:
        return zeros(3)
    return (d / (distance**3)) * (m1 * m2)

######### Node class (fixed) #######
class Node:
    def __init__(self, m, x, y, z):
        self.m = m
        self.m_pos = m * array([x, y, z], dtype=np_float64)  # Use numpy's float64
        self.momentum = zeros(3, dtype=np_float64)
        self.child = None
        self.s = 1.0
        self.relpos = self.pos().copy()

    def pos(self):
        return self.m_pos / self.m

    def into_next_quadrant(self):
        self.s *= 0.5
        x_quad = self._subdivide(0)
        y_quad = self._subdivide(1)
        z_quad = self._subdivide(2)
        return x_quad + 2 * y_quad + 4 * z_quad

    def _subdivide(self, i):
        self.relpos[i] *= 2.0
        quadrant = 0 if self.relpos[i] < 1.0 else 1
        if quadrant == 1:
            self.relpos[i] -= 1.0
        return quadrant

    def reset_to_0th_quadrant(self):
        self.s = 1.0
        self.relpos = self.pos().copy()

    def dist(self, other):
        return norm(other.pos() - self.pos())

######### Tree functions #######
def add(body, node):
    smallest_quadrant = 1.e-4
    if node is None:
        return body

    if node.s > smallest_quadrant:
        if node.child is None:
            new_node = Node(node.m, 0, 0, 0)
            new_node.m_pos = node.m_pos.copy()
            new_node.momentum = node.momentum.copy()
            new_node.child = [None] * 8
            quadrant = node.into_next_quadrant()
            new_node.child[quadrant] = node
            node = new_node

        node.m += body.m
        node.m_pos += body.m_pos
        quadrant = body.into_next_quadrant()
        node.child[quadrant] = add(body, node.child[quadrant])

    return node

def force_on(body, root, theta):
    force = zeros(3)
    stack = [root]
    body_pos = body.pos()
    body_m = body.m

    while stack:
        node = stack.pop()
        if node.child is None:
            force += compute_force(node.pos(), body_pos, node.m, body_m)
        else:
            if node.s < theta * node.dist(body):
                force += compute_force(node.pos(), body_pos, node.m, body_m)
            else:
                stack.extend([c for c in node.child if c is not None])
    return force

######### Simulation loop #######
def verlet(bodies, root, theta, G, dt):
    for body in bodies:
        f = G * force_on(body, root, theta)
        body.momentum += dt * f
        body.m_pos += dt * body.momentum

def plot_bodies(bodies, i):
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.scatter([b.pos()[0] for b in bodies],
               [b.pos()[1] for b in bodies],
               [b.pos()[2] for b in bodies], s=1)
    ax.set_xlim([0., 1.0])
    ax.set_ylim([0., 1.0])
    ax.set_zlim([0., 1.0])
    plt.gcf().savefig(f'bodies3D_{i:06}.png')
    plt.close()

######### Main program #######
theta = 0.5
mass = 1.2
ini_radius = 0.1
inivel = 0.1
G = 6.e-6
dt = 1.e-3
numbodies = 250
max_iter = 100000
img_iter = 20

random.seed(1)
posx = random.random(numbodies) * 2 * ini_radius + 0.5 - ini_radius
posy = random.random(numbodies) * 2 * ini_radius + 0.5 - ini_radius
posz = random.random(numbodies) * 2 * ini_radius + 0.5 - ini_radius

# Filter bodies within sphere
bodies = []
for px, py, pz in zip(posx, posy, posz):
    if (px-0.5)**2 + (py-0.5)**2 + (pz-0.5)**2 < ini_radius**2:
        bodies.append(Node(mass, px, py, pz))

# Initialize momentum in x-y plane
for body in bodies:
    r = body.pos() - array([0.5, 0.5, body.pos()[2]])
    body.momentum = array([-r[1], r[0], 0.]) * mass * inivel * norm(r) / ini_radius

# Main loop
for i in range(max_iter):
    root = None
    for body in bodies:
        body.reset_to_0th_quadrant()
        root = add(body, root)
    print(f"Iteration {i}: value = {bodies[0].pos()[2]}")
    verlet(bodies, root, theta, G, dt)
    if i % img_iter == 0:
        plot_bodies(bodies, i // img_iter)