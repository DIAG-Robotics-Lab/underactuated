import numpy as np
import math
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
alpha_0 = 1
alpha = alpha_0

x_ini = np.array([0, 0, 0, 0])
x_ter = np.array([math.pi, 0, 0, 0])

x = np.zeros((n, N+1))
x[:, 0] = x_ini

opt_sym = cs.Opti()
X_ = opt_sym.variable(4)
U_ = opt_sym.variable(1)

#ff, p = model.get_cart_pendulum_model()
ff, p = model.get_pendubot_model()
f_ = lambda x, u: x + delta * ff(x, u)

f = cs.Function('f', [X_, U_], [f_(X_,U_)], {"post_expand": True})
fx = cs.Function('fx', [X_, U_], [cs.jacobian(f_(X_,U_), X_)], {"post_expand": True})
fu = cs.Function('fu', [X_, U_], [cs.jacobian(f_(X_,U_), U_)], {"post_expand": True})

u = np.ones(N) * 1
for i in range(N):
  x[:, i+1] = np.array(f(x[:,i], u[i])).flatten()

total_time = 0

# optimization problem
opt = cs.Opti('conic')
opt.solver('proxqp')

X = opt.variable(4,N+1)
U = opt.variable(1,N)
X_guess = opt.parameter(4,N+1)
U_guess = opt.parameter(1,N)

opt.subject_to( X[:,0] == x_ini )
opt.subject_to( X[:,N] == x_ter )
for i in range(N):
  opt.subject_to( X[:,i+1] == f(X_guess[:,i], U_guess[:,i]) + \
                  fx(X_guess[:,i], U_guess[:,i]) @ (X[:,i] - X_guess[:,i]) + \
                  fu(X_guess[:,i], U_guess[:,i]) @ (U[:,i] - U_guess[:,i]) )
  #X[:,i+1] = f(X_guess[:,i], U_guess[:,i]) + \
  #           fx(X_guess[:,i], U_guess[:,i]) @ (X[:,i] - X_guess[:,i]) + \
  #           fu(X_guess[:,i], U_guess[:,i]) @ (U[:,i] - U_guess[:,i])

cost = (x_ter - X[:,N]).T @ Qter @ (x_ter - X[:,N])
for i in range(N):
  cost = cost + (x_ter - X[:,i]).T @ Q @ (x_ter - X[:,i]) + U[:,i].T @ R @ U[:,i]

opt.minimize(cost)

for iter in range(iterations):
  start_time = time.time()

  opt.set_value(X_guess, x)
  opt.set_value(U_guess, u)

  sol = opt.solve()
  u = sol.value(U)
  x = sol.value(X)

  elapsed_time = time.time() - start_time
  total_time += elapsed_time
  print('Iteration time: ', elapsed_time*1000, ' ms')

xcheck = np.zeros((n, N+1))
xcheck[:,0] = x_ini
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:,i], u[i])).flatten()

# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(N, xcheck, u, p)


