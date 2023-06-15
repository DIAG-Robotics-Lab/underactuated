import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin
from math import cos
from casadi import *

# parameters
N = 1000
delta = 0.01
l = 1
m1 = 10
m2 = 5
g = 9.81
b1 = 0
b2 = 0

# state
q1 = 0
q2 = 0
q1_d = 0
q2_d = 0

# trajectories
u_t = np.zeros(N)
q1_t = np.zeros(N+1)
q2_t = np.zeros(N+1)
q1_t[0] = q1
q2_t[0] = q2
x = np.zeros((4,N+1))

# display
sz = l/4

# dynamics
f1 = lambda q1, q2, q1_d, q2_d, u: (l*m2*sin(q2)*q2_d**2 + u + m2*g*cos(q2)*sin(q2)) / (m1 + m2*(1-cos(q2)**2)) - b1*q1_d
f2 = lambda q1, q2, q1_d, q2_d, u: - (l*m2*cos(q2)*sin(q2)*q2_d**2 + u*cos(q2) + (m1+m2)*g*sin(q2)) / (l*m1 + l*m2*(1-cos(q2)**2)) - b2*q2_d

# casadi nonlinear optimization
opti = Opti()
X = opti.variable(4,N+1)
U = opti.variable(1,N)
cost = 0

for i in range(N):
    opti.subject_to( X[:,i+1] == X[:,i] + delta * vertcat(X[2:4,i], f1(X[0,i], X[1,i], X[2,i], X[3,i], U[0,i]), f2(X[0,i], X[1,i], X[2,i], X[3,i], U[0,i])) )
    cost += U[0,i]**2

opti.subject_to( X[:,0] == (q1, q2, q1_d, q2_d) )
opti.subject_to( X[1:4,N] == (math.pi, 0, 0) )

opti.minimize(cost)

opti.solver("ipopt")
sol = opti.solve()

u_t = sol.value(U)

# integrate
for i in range(N):
    u = u_t[i]
    q1_dd = f1(q1, q2, q1_d, q2_d, u)
    q2_dd = f2(q1, q2, q1_d, q2_d, u)
    q1_d += q1_dd * delta
    q2_d += q2_dd * delta
    q1 += q1_d * delta;
    q2 += q2_d * delta;
    
    q1_t[i+1] = q1
    q2_t[i+1] = q2
    
q1_t = sol.value(X)[0,:]
q2_t = sol.value(X)[1,:]

# display
def animate(i):
    plt.clf()
    plt.xlim((-5*l, 5*l))
    plt.ylim((-1.5*l, 1.5*l))
    plt.gca().set_aspect('equal')
    
    plt.plot(np.array((q1_t[i] + sz, q1_t[i] + sz, q1_t[i] - sz, q1_t[i] - sz, q1_t[i] + sz)),  np.array((sz, -sz, -sz, sz, sz)))
    plt.gca().add_patch(plt.Circle((q1_t[i]+math.sin(q2_t[i]), -math.cos(q2_t[i])), sz/2, color='blue'))
    plt.plot(np.array((q1_t[i], q1_t[i]+math.sin(q2_t[i]))), np.array((0, -math.cos(q2_t[i]))))

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)

plt.show()
