import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin
from math import cos
from casadi import *
from qpsolvers import solve_qp

# parameters
N = 200
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
X = topt.variable(4,N+1)
U = topt.variable(1,N)
cost = 0

for i in range(N):
    topt.subject_to( X[:,i+1] == X[:,i] + delta * vertcat(X[2:4,i], f1(X[:,i], U[0,i]), f2(X[:,i], U[0,i])) )
    cost += U[0,i]**2

topt.subject_to( X[:,0] == x[:, 0] )
topt.subject_to( X[1,N] == math.pi )
topt.subject_to( X[2:4,N] == (0, 0) )

topt.minimize(cost)

topt.solver("ipopt")
sol = topt.solve()
u = sol.value(U)

"""
# sequential quadratic programming
u = np.ones(N)
x = np.zeros((4,N+1))
X_ = MX.sym("X", 4, 1)
U_ = MX.sym("U", 1, 1)

T = np.zeros((4*N,N))
S = np.zeros((4*N,4))

df1_dx = Function('df1_dx', [X_,U_], [jacobian(f1(X_,U_), X_)])
df1_du = Function('df1_du', [X_,U_], [jacobian(f1(X_,U_), U_)])
df2_dx = Function('df2_dx', [X_,U_], [jacobian(f2(X_,U_), X_)])
df2_du = Function('df2_du', [X_,U_], [jacobian(f2(X_,U_), U_)])

N_SQP = 10
for j in range(N_SQP):
    print(j)
    for i in range(N):
        A = np.identity(4) + delta * np.vstack((np.array((0,0,1,0)), np.array((0,0,0,1)), df1_dx(x[:,i], u[i]), df2_dx(x[:,i], u[i])))
        B = delta * np.vstack(((0), (0), df1_du(x[:,i], u[i]), df2_du(x[:,i], u[i])))

        if i > 0:
            T[4*i:4*(i+1),i:i+1] = B
            T[4*i:4*(i+1),0:i] = A @ T[4*(i-1):4*i,0:i]
            S[4*i:4*(i+1),:] = A @ S[4*(i-1):4*i,:]
        else:
            S[4*i:4*(i+1),:] = A
            
        x[:,i+1] = x[:,i] + delta * np.hstack((x[2:4,i], f1(x[:,i], u[i]), f2(x[:,i], u[i])))

    H = np.identity(N)
    f = np.zeros(N)    
    A_eq = T[4*(N-1)+1:4*N,:]
    b_eq = - S[4*(N-1)+1:4*N,:] @ x[:,0] + np.array((math.pi, 0, 0))
    A_ineq = np.zeros((1,N))
    b_ineq = np.zeros((1,1))
        
    u = solve_qp(H, f, A_eq, b_eq, A_ineq, b_ineq, solver="qpoases")
 
    plt.clf()
    plt.plot(x[0,:]) 
    plt.savefig('traj' + str(j) + '.png')
"""

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
