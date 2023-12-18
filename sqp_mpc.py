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
N_sim = 200
Q = np.eye(n)
R = np.eye(m)
Qter = np.eye(n)

x_ini = np.array([0, 0, 0, 0])
x_ter = np.array([math.pi, 0, 0, 0])
#x_ter = np.array([0, math.pi, 0, 0])

u = np.zeros(N_sim)
x = np.zeros((n, N_sim+1))
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

total_time = 0

# optimization problem
opt = cs.Opti('conic')
opt.solver('proxqp')

X = opt.variable(4,N+1)
U = opt.variable(1,N)
X_guess = opt.parameter(4,N+1)
U_guess = opt.parameter(1,N)
x0_param = opt.parameter(4)

opt.subject_to( X[:,0] == x0_param )
opt.subject_to( X[:,N] == x_ter )
for i in range(N):
  opt.subject_to( X[:,i+1] == f(X_guess[:,i], U_guess[i]) + \
                  fx(X_guess[:,i], U_guess[i]) @ (X[:,i] - X_guess[:,i]) + \
                  fu(X_guess[:,i], U_guess[i]) @ (U[:,i] - U_guess[i]) )

cost = (x_ter - X[:,N]).T @ Qter @ (x_ter - X[:,N])
for i in range(N):
  cost = cost + (x_ter - X[:,i]).T @ Q @ (x_ter - X[:,i]) + U[:,i].T @ R @ U[:,i]

opt.minimize(cost)

u_pred = np.ones(N) * 0.0
x_pred = np.zeros((n,N+1))
for i in range(N):
  x_pred[:, i+1] = np.array(f(x[:,i], u[i])).flatten()

for j in range(N_sim):
  start_time = time.time()

  opt.set_value(X_guess, x_pred)
  opt.set_value(U_guess, u_pred)
  opt.set_value(x0_param, x[:,j])

  sol = opt.solve()
  u_pred = sol.value(U)
  x_pred = sol.value(X)

  u[j] = u_pred[0]

  x_pred[:,0:N] = x_pred[:,1:N+1]
  x_pred[:,N] = x_pred[:,N-1]

  u_pred[0:N-1] = u_pred[1:N]
  u_pred[N-1] = u_pred[N-2]

  x[:,j+1] = np.array(f(x[:,j], u[j])).flatten()

  elapsed_time = time.time() - start_time
  total_time += elapsed_time
  print('Iteration time: ', elapsed_time*1000, ' ms')

# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(N_sim, x, u, p)


