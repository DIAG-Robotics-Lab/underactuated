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
N_sim = 200
n, m = mod.n, mod.m
delta = 0.01
Q = np.eye(mod.n)
R = np.eye(mod.m)
Qter = np.eye(mod.n)
max_iters = 10
x_init = np.zeros(n)
if   mod.name == 'cart_pendulum': x_goal = np.array([0, cs.pi, 0, 0])
elif mod.name == 'pendubot'     : x_goal = np.array([cs.pi, 0, 0, 0])
elif mod.name == 'uav'          : x_goal = np.array([1, 1, 0, 0, 0, 0])

# trajectory
x = np.zeros((n, N_sim+1))
u = np.zeros((m, N_sim))
x[:, 0] = x_init

# dynamics and its derivatives
f_ = lambda x, u: x + delta * mod.f(x, u)
X_ = opt.variable(mod.n)
U_ = opt.variable(mod.m)
f  = cs.Function('f' , [X_,U_], [f_(X_,U_)]                , {"post_expand": True})
fx = cs.Function('fx', [X_,U_], [cs.jacobian(f_(X_,U_),X_)], {"post_expand": True})
fu = cs.Function('fu', [X_,U_], [cs.jacobian(f_(X_,U_),U_)], {"post_expand": True})

# optimization problem
opt = cs.Opti('conic')
opt.solver('proxqp')

X = opt.variable(mod.n,N+1)
U = opt.variable(mod.m,N)
X_guess = opt.parameter(mod.n,N+1)
U_guess = opt.parameter(mod.m,N)
x0_param = opt.parameter(mod.n)

opt.subject_to( X[:,0] == x0_param )
opt.subject_to( X[:,N] == x_goal )
for i in range(N):
  opt.subject_to( X[:,i+1] == f(X_guess[:,i], U_guess[i]) + \
                  fx(X_guess[:,i], U_guess[i]) @ (X[:,i] - X_guess[:,i]) + \
                  fu(X_guess[:,i], U_guess[i]) @ (U[:,i] - U_guess[:,i]) )
if mod.name == 'uav'          :
  opt.subject_to( X[:,N] == (1, 1, 0, 0, 0, 0) )
  for i in range(N):
    opt.subject_to( U[0,i] >= 0 )
    opt.subject_to( U[1,i] >= 0 )

cost = (x_goal - X[:,N]).T @ Qter @ (x_goal - X[:,N])
for i in range(N):
  cost = cost + (x_goal - X[:,i]).T @ Q @ (x_goal - X[:,i]) + U[:,i].T @ R @ U[:,i]

opt.minimize(cost)

u_pred = np.ones((m,N)) * 0.0
x_pred = np.zeros((n,N+1))
for i in range(N):
  x_pred[:, i+1] = np.array(f(x[:,i], u[:,i])).flatten()

x_pred_record = []

for j in range(N_sim):
  start_time = time.time()

  opt.set_value(X_guess, x_pred)
  opt.set_value(U_guess, u_pred)
  opt.set_value(x0_param, x[:,j])

  # solve QP and integrate
  sol = opt.solve()
  u_pred = np.asmatrix(sol.value(U))
  x_pred = np.asmatrix(sol.value(X))
  x_pred_record.append(x_pred)

  # set initial guess for next iteration
  opt.set_initial(U, u_pred)
  opt.set_initial(X, x_pred)

  u[:,j] = u_pred[:,0]
  x[:,j+1] = np.array(f(x[:,j], u[:,j])).flatten()

  elapsed_time = time.time() - start_time
  print('Iteration time: ', elapsed_time * 1000, ' ms')

# display
#mod.animate(N_sim, x, u)
mod.animate(N_sim, x, u, x_pred=x_pred_record)


