import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import time
import model

# initialization
mod = model.Pendubot()
Δ = 0.01
n, m = mod.n, mod.m
N = 100
max_ddp_iters = 10
max_line_search_iters = 10
Q = np.eye(n) * 0
R = np.eye(m) * 0.01
Q_ter = np.eye(n) * 10000

if   mod.name == 'cart_pendulum': x_ter = np.array((0, cs.pi, 0, 0))
elif mod.name == 'pendubot'     : x_ter = np.array((cs.pi, 0, 0, 0))
elif mod.name == 'uav'          : x_ter = np.array((1, 1, 0, 0, 0, 0))

# symbolic variables
opt = cs.Opti()
X = opt.variable(n)
U = opt.variable(m)

# cost function
L_ = lambda x, u: (x_ter - x).T @ Q @ (x_ter - x) + u.T @ R @ u
L_ter_ = lambda x: (x_ter - x).T @ Q_ter @ (x_ter - x)
L       = cs.Function('L'      , [X, U], [L_(X,U)]                  , {"post_expand": True})
L_ter   = cs.Function('L_ter'  , [X]   , [L_ter_(X)]                , {"post_expand": True})
Lx      = cs.Function('Lx'     , [X, U], [cs.jacobian(L(X,U), X)]   , {"post_expand": True})
Lu      = cs.Function('Lu'     , [X, U], [cs.jacobian(L(X,U), U)]   , {"post_expand": True})
Lxx     = cs.Function('Lxx'    , [X, U], [cs.jacobian(Lx(X,U), X)]  , {"post_expand": True})
Lux     = cs.Function('Lux'    , [X, U], [cs.jacobian(Lu(X,U), X)]  , {"post_expand": True})
Luu     = cs.Function('Luu'    , [X, U], [cs.jacobian(Lu(X,U), U)]  , {"post_expand": True})
L_terx  = cs.Function('L_terx' , [X]   , [cs.jacobian(L_ter(X), X)] , {"post_expand": True})
L_terxx = cs.Function('L_terxx', [X]   , [cs.jacobian(L_terx(X), X)], {"post_expand": True})

# dynamics
f_cont = mod.f
f_ = lambda x, u: x + Δ * f_cont(x, u)
f = cs.Function('f', [X, U], [f_(X,U)], {"post_expand": True})
fx = cs.Function('fx', [X, U], [cs.jacobian(f_(X,U), X)], {"post_expand": True})
fu = cs.Function('fu', [X, U], [cs.jacobian(f_(X,U), U)], {"post_expand": True})

# initial forward pass
x = np.zeros((n, N+1))
u = np.ones((m, N))
x[:, 0] = np.zeros(n)

cost = 0
for i in range(N):
  x[:,i+1] = np.array(f(x[:,i], u[:,i])).flatten()
  cost += L(x[:,i], u[:,i])
cost += L_ter(x[:, N])

k = [np.zeros((m,1))] * (N+1)
K = [np.zeros((m,n))] * (N+1)

V = np.zeros(N+1)
Vx = np.zeros((n,N+1))
Vxx = np.zeros((n,n,N+1))

total_time = 0

for iter in range(max_ddp_iters):
  # backward pass
  backward_pass_start_time = time.time()
  V[N] = L_ter(x[:,N])
  Vx[:,N] = L_terx(x[:,N])
  Vxx[:,:,N] = L_terxx(x[:,N])

  for i in reversed(range(N)):
    fx_eval = fx(x[:,i], u[:,i])
    fu_eval = fu(x[:,i], u[:,i])

    Qx = Lx(x[:,i], u[:,i]).T + fx_eval.T @ Vx[:,i+1]
    Qu = Lu(x[:,i], u[:,i]).T + fu_eval.T @ Vx[:,i+1]

    Qxx = Lxx(x[:,i], u[:,i]) + fx_eval.T @ Vxx[:,:,i+1] @ fx_eval
    Quu = Luu(x[:,i], u[:,i]) + fu_eval.T @ Vxx[:,:,i+1] @ fu_eval
    Qux = Lux(x[:,i], u[:,i]) + fu_eval.T @ Vxx[:,:,i+1] @ fx_eval

    Quu_inv = np.linalg.inv(Quu)
    k[i] = - Quu_inv @ Qu
    K[i] = - Quu_inv @ Qux

    V[i] = V[i+1] - 0.5 * k[i].T @ Quu @ k[i]
    Vx[:,i] = np.array(Qx - K[i].T @ Quu @ k[i]).flatten()
    Vxx[:,:,i] = Qxx - K[i].T @ Quu @ K[i]

  backward_pass_time = time.time() - backward_pass_start_time

  # forward pass
  forward_pass_start_time = time.time()
  unew = np.ones((m, N))
  xnew = np.zeros((n, N+1))
  xnew[:,0] = x[:,0]

  # line search
  alpha = 1.
  for ls_iter in range(max_line_search_iters):
    new_cost = 0
    for i in range(N):
      unew[:,i] = np.array(u[:,i] + alpha * k[i] + K[i] @ (xnew[:,i] - x[:,i])).flatten()
      xnew[:,i+1] = np.array(f(xnew[:,i], unew[:,i])).flatten()
      new_cost = new_cost + L(xnew[:,i], unew[:,i])
    new_cost = new_cost + L_ter(xnew[:,N])

    if new_cost < cost:
      cost = new_cost
      x = xnew
      u = unew
      break
    else:
      alpha /= 2.

  forward_pass_time = time.time() - forward_pass_start_time
  total_time += backward_pass_time + forward_pass_time
  print('Iteration:', iter, 'BP Time:', round(backward_pass_time*1000), 'FP Time:', round(forward_pass_time*1000))

print('Total time: ', total_time*1000, ' ms')

# check result
xcheck = np.zeros((n, N+1))
xcheck[:,0] = np.zeros(n)
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:,i], u[:,i])).flatten()

# display
mod.animate(N, xcheck, u)