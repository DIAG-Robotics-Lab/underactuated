import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model
import animation

delta = 0.01
n, m = (4, 1)
N = 100
Q = np.eye(n)
R = np.eye(m)
Qter = np.eye(n)
iterations = 100

x_init = np.array([0, 0, 0, 0])
x_goal = np.array([cs.pi, 0, 0, 0])

x = np.zeros((n, N+1))
x[:, 0] = x_init

opt_sym = cs.Opti()
X_ = opt_sym.variable(4)
U_ = opt_sym.variable(1)

#ff, p = model.get_cart_pendulum_model()
ff, p = model.get_pendubot_model()
f_ = lambda x, u: x + delta * ff(x, u)

f  = cs.Function('f' , [X_, U_], [f_(X_,U_)]                 , {"post_expand": True})
fx = cs.Function('fx', [X_, U_], [cs.jacobian(f_(X_,U_), X_)], {"post_expand": True})
fu = cs.Function('fu', [X_, U_], [cs.jacobian(f_(X_,U_), U_)], {"post_expand": True})

u = np.ones(N) * 1
for i in range(N):
  x[:, i+1] = np.array(f(x[:,i], u[i])).flatten()

total_time = 0

# optimization problem
opt = cs.Opti('conic')
opt.solver('proxqp')

dX = opt.variable(4,N+1)
dU = opt.variable(1,N)
X = opt.parameter(4,N+1)
U = opt.parameter(1,N)

opt.subject_to( dX[:,0] == (0,0,0,0) )
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

for iter in range(iterations):
  start_time = time.time()

  opt.set_value(X, x)
  opt.set_value(U, u)

  sol = opt.solve()
  u = sol.value(U) + sol.value(dU)
  x = sol.value(X) + sol.value(dX)

  elapsed_time = time.time() - start_time
  total_time += elapsed_time
  print('Iteration time: ', elapsed_time*1000, ' ms')

xcheck = np.zeros((n, N+1))
xcheck[:,0] = x_init
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:,i], u[i])).flatten()

# display
animation.animate_pendubot(N, xcheck, u, p)


