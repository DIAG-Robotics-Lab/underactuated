import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin
from math import cos
from casadi import *
from qpsolvers import solve_qp

# parameters
M = 10
num = 20
N = M*num
delta = 0.01
l = 1
m1 = 10
m2 = 5
g = 9.81
b1 = 0
b2 = 0

# trajectory
u = np.zeros(N)
x = np.zeros((4,N+1))

# display
sz = l/4
np.set_printoptions(precision=2)

# dynamics
f1 = lambda x, u: (l*m2*sin(x[1])*x[3]**2 + u + m2*g*cos(x[1])*sin(x[1])) / (m1 + m2*(1-cos(x[1])**2)) - b1*x[2]
f2 = lambda x, u: - (l*m2*cos(x[1])*sin(x[1])*x[3]**2 + u*cos(x[1]) + (m1+m2)*g*sin(x[1])) / (l*m1 + l*m2*(1-cos(x[1])**2)) - b2*x[3]

# casadi nonlinear toptmization
topt = Opti()
X = []
U = topt.variable(1,N)
cost = 0

for j in range(num):
    X.append(topt.variable(4,M+1))
    for i in range(M):
        X[j][:,i+1] = X[j][:,i] + delta * vertcat(X[j][2:4,i], f1(X[j][:,i], U[0,j*M+i]), f2(X[j][:,i], U[0,j*M+i]))
        
for j in range(num-1):
    topt.subject_to( X[j][:,M] == X[j+1][:,0] )
        
for i in range(N):
    cost += U[0,i]**2

topt.subject_to( X[0][:,0] == x[:, 0] )
topt.subject_to( X[num-1][1,M] == math.pi )
topt.subject_to( X[num-1][2:4,M] == (0, 0) )

topt.minimize(cost)

topt.solver("ipopt")
sol = topt.solve()
u = sol.value(U)

# integrate
for i in range(N):
    x[:,i+1] = x[:,i] + delta * np.hstack((x[2:4,i], f1(x[:,i], u[i]), f2(x[:,i], u[i])))

# display
def animate(i):
    plt.clf()
    plt.xlim((-5*l, 5*l))
    plt.ylim((-1.5*l, 1.5*l))
    plt.gca().set_aspect('equal')
    
    plt.plot(x[0,:][i] + np.array((sz, sz, - sz, - sz, + sz)),  np.array((sz, -sz, -sz, sz, sz)))
    plt.gca().add_patch(plt.Circle((x[0,:][i] + sin(x[1,:][i]), - cos(x[1,:][i])), sz/2, color='blue'))
    plt.plot(np.array((x[0,:][i], x[0,:][i] + sin(x[1,:][i]))), np.array((0, - cos(x[1,:][i]))))

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)

plt.show()
