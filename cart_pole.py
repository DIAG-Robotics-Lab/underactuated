import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin
from math import cos

# parameters
dur = 2000
delta = 0.01
l = 1
m1 = 10
m2 = 5
g = 9.81
b1 = 1
b2 = 1

# state
q1 = 0
q2 = math.pi/2
q1_d = 0
q2_d = 0

# trajectories
u_t = np.zeros(dur)
q1_t = np.zeros(dur)
q2_t = np.zeros(dur)
q1_t[0] = q1
q2_t[0] = q2

# display
sz = l/4

# dynamics
f1 = lambda q1, q2, q1_d, q2_d, u: (l*m2*sin(q2)*q2_d**2 + u + m2*g*cos(q2)*sin(q2)) / (m1 + m2*(1-cos(q2)**2)) - b1*q1_d
f2 = lambda q1, q2, q1_d, q2_d, u: - (l*m2*cos(q2)*sin(q2)*q2_d**2 + u*cos(q2) + (m1+m2)*g*sin(q2)) / (l*m1 + l*m2*(1-cos(q2)**2)) - b2*q2_d

for i in range(1, dur):
    u = u_t[i]
    q1_dd = f1(q1, q2, q1_d, q2_d, u)
    q2_dd = f2(q1, q2, q1_d, q2_d, u)
    q1_d += q1_dd*delta
    q2_d += q2_dd*delta
    q1 += q1_d*delta + 0.5*q1_dd*delta**2
    q2 += q2_d*delta + 0.5*q2_dd*delta**2
    
    q1_t[i] = q1
    q2_t[i] = q2

def animate(i):
    plt.clf()
    plt.xlim((-2*l, 2*l))
    plt.ylim((-1.5*l, 1.5*l))
    plt.gca().set_aspect('equal')
    
    plt.plot(np.array((q1_t[i] + sz, q1_t[i] + sz, q1_t[i] - sz, q1_t[i] - sz, q1_t[i] + sz)),  np.array((sz, -sz, -sz, sz, sz)))
    circle = plt.Circle((q1_t[i]+math.sin(q2_t[i]), -math.cos(q2_t[i])), sz/2, color='blue')
    plt.gca().add_patch(circle)
    graph = plt.plot(np.array((q1_t[i], q1_t[i]+math.sin(q2_t[i]))),  -np.array((0, math.cos(q2_t[i]))))
    return graph

ani = FuncAnimation(plt.gcf(), animate, frames=dur, repeat=False, interval=0)
plt.show()
