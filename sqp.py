import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model

# initialization
opt = cs.Opti('conic')
opt.solver('proxqp')
mod = model.Pendubot()
N = 100
delta = 0.01
Q = np.eye(mod.n)
R = np.eye(mod.m)
Qter = np.eye(mod.n)
max_iters = 20
x_init = np.zeros(mod.n)

if   mod.name == 'cart_pendulum': x_goal = np.array((0, cs.pi, 0, 0))
elif mod.name == 'pendubot'     : x_goal = np.array((cs.pi, 0, 0, 0))
elif mod.name == 'uav'          : x_goal = np.array((1, 1, 0, 0, 0, 0))

# initial guess
x = np.zeros((mod.n,N+1))
u = np.zeros((mod.m,N))
x[:, 0] = x_init

# dynamics and its derivatives
f_ = lambda x, u: x + delta * mod.f(x, u)
X_ = opt.variable(mod.n)
U_ = opt.variable(mod.m)
f  = cs.Function('f' , [X_,U_], [f_(X_,U_)]                , {"post_expand": True})
fx = cs.Function('fx', [X_,U_], [cs.jacobian(f_(X_,U_),X_)], {"post_expand": True})
fu = cs.Function('fu', [X_,U_], [cs.jacobian(f_(X_,U_),U_)], {"post_expand": True})

# optimization problem
dX = opt.variable(mod.n,N+1)
dU = opt.variable(mod.m,N)
X = opt.parameter(mod.n,N+1)
U = opt.parameter(mod.m,N)

opt.subject_to( dX[:,0] == np.zeros(mod.n) )
opt.subject_to( X[:,N] + dX[:,N] == x_goal )
for i in range(N):
  opt.subject_to( X[:,i+1] + dX[:,i+1] == f(X[:,i], U[:,i]) + \
                  fx(X[:,i], U[:,i]) @ dX[:,i] + \
                  fu(X[:,i], U[:,i]) @ dU[:,i] )

cost = (X[:,N] + dX[:,N] - x_goal).T @ Qter @ (X[:,N] + dX[:,N] - x_goal)
for i in range(N):
  cost = cost + \
         (X[:,i] + dX[:,i] - x_goal).T @ Q @ (X[:,i] + dX[:,i] - x_goal) + \
         (U[:,i] + dU[:,i]).T @ R @ (U[:,i] + dU[:,i])

opt.minimize(cost)

# SQP iterations
for iter in range(max_iters):
  start_time = time.time()

  opt.set_value(X, x)
  opt.set_value(U, u)

  sol = opt.solve()
  x = np.asmatrix(sol.value(X) + sol.value(dX))
  u = np.asmatrix(sol.value(U) + sol.value(dU))

  elapsed_time = time.time() - start_time
  print('Iteration time: ', elapsed_time * 1000, ' ms')

xcheck = np.zeros((mod.n,N+1))
xcheck[:,0] = x_init
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:,i], u[:,i])).flatten()

# display
mod.animate(N, xcheck, u)