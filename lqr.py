import numpy as np
import scipy
import math
import casadi as cs
import animation
import model

opt = cs.Opti()

# parameters
N = 200
delta = 0.01
g = 9.81

# trajectory
u = np.zeros(N)
x = np.zeros((4,N+1))
x[0,0] = math.pi + 0.2

# generate model
#f, p = model.get_cart_pendulum_model()
f, p = model.get_pendubot_model()

X = opt.variable(4)
U = opt.variable(1)

fx = cs.Function('fx', [X, U], [cs.jacobian(f(X,U), X)])
fu = cs.Function('fu', [X, U], [cs.jacobian(f(X,U), U)])

# solve LQR
A = np.array(fx([math.pi, 0, 0, 0], 0))
B = np.array(fu([math.pi, 0, 0, 0], 0))
R = np.identity(1)
Q = np.identity(4)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
print(P @ A + A.T @ P - P @ B @ np.linalg.inv(R) @ B.T @ P + Q)

# integrate
for i in range(N):
  u[i] = - np.linalg.inv(R) @ B.T @ P @ (x[:,i] - [math.pi, 0, 0, 0])
  x[:,i+1] = x[:,i] + delta * f(x[:,i], u[i]).full().squeeze()
  
# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(N, x, u, p)
