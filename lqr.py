import numpy as np
import scipy
import casadi as cs
import model

# initialization
opt = cs.Opti()
mod = model.Pendubot()
N = 200
delta = 0.01
f = mod.f

# trajectory
x = np.zeros((mod.n,N+1))
u = np.zeros((mod.m,N))
x[:,0] = (cs.pi + 0.2, 0, 0, 0)

# compute derivatives
X = opt.variable(4)
U = opt.variable(1)
fx = cs.Function('fx', [X, U], [cs.jacobian(f(X,U), X)])
fu = cs.Function('fu', [X, U], [cs.jacobian(f(X,U), U)])

# solve LQR
A = np.array(fx([cs.pi, 0, 0, 0], 0))
B = np.array(fu([cs.pi, 0, 0, 0], 0))
R = np.identity(1)
Q = np.identity(4)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
print(P @ A + A.T @ P - P @ B @ np.linalg.inv(R) @ B.T @ P + Q)

# integrate
for i in range(N):
  u[:,i] = - np.linalg.inv(R) @ B.T @ P @ (x[:,i] - [cs.pi, 0, 0, 0])
  x[:,i+1] = x[:,i] + delta * f(x[:,i], u[:,i]).full().squeeze()
  
# display
mod.animate(N, x, u)