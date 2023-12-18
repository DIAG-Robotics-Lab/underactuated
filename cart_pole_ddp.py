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
Q_ter = np.eye(n) * 10000
Qter = 1000 * np.eye(n)
iterations = 20
g = 9.81

alpha_0 = 0.9

x = np.zeros((n, N+1))
x[:, 0] = np.array([0, 0, 0, 0])

opt = cs.Opti()
X = opt.variable(4)
U = opt.variable(1)

x_ter = np.array((math.pi, 0, 0, 0))
L_ = lambda x, u: (x_ter - x).T @ Q @ (x_ter - x) + u.T @ R @ u
#L_ter_ = lambda x_ter: 1000 * ((x_ter[1] - math.pi)**2 + x_ter[2]**2 + x_ter[3]**2)
#L_ter_ = lambda x_ter: 10000 * ((x_ter[0] - math.pi)**2 + x_ter[1]**2 + x_ter[2]**2 + x_ter[3]**2)
L_ter_ = lambda x: (x_ter - x).T @ Q_ter @ (x_ter - x)

#ff, p = model.get_cart_pendulum_model()
ff, p = model.get_pendubot_model()
f_ = lambda x, u: x + delta * ff(x, u)

f = cs.Function('f', [X, U], [f_(X,U)], {"post_expand": True})
L = cs.Function('L', [X, U], [L_(X,U)], {"post_expand": True})
L_ter = cs.Function('L_ter', [X], [L_ter_(X)], {"post_expand": True})
fx = cs.Function('fx', [X, U], [cs.jacobian(f_(X,U), X)], {"post_expand": True})
fu = cs.Function('fu', [X, U], [cs.jacobian(f_(X,U), U)], {"post_expand": True})
#Lx = cs.Function('Lx', [X, U], [cs.jacobian(L_(X,U), X)])
#Lu = cs.Function('Lu', [X, U], [cs.jacobian(L_(X,U), U)])
#Lxx = cs.Function('Lxx', [X, U], [cs.jacobian(cs.jacobian(L_(X,U), X).T, X)])
#Lux = cs.Function('Lux', [X, U], [cs.jacobian(cs.jacobian(L_(X,U), U).T, X)])
#Luu = cs.Function('Luu', [X, U], [cs.jacobian(cs.jacobian(L_(X,U), U).T, U)])
L_terx = cs.Function('L_terx', [X], [cs.jacobian(L_ter_(X), X)], {"post_expand": True})
L_terxx = cs.Function('L_terxx', [X], [cs.jacobian(cs.jacobian(L_ter_(X), X), X)], {"post_expand": True})

u = np.array([1 * np.ones(N)])

cost = 0
for i in range(N):
  x[:, i+1] = np.array(f(x[:,i], u[:,i])).flatten()
  cost = cost + L(x[:,i], u[:,i])

cost = cost + L_ter(x[:, N])

k = [np.array((m, 1))] * (N+1)
K = [np.array((m, n))] * (N+1)

V = np.zeros(N+1)
Vx = np.zeros((n,N+1))
Vxx = np.zeros((n,n,N+1))

xnew = np.zeros((n, N+1))

total_time = 0
update_magnitude = np.zeros(iterations)

for iter in range(iterations):
  start_time = time.time()

  alpha = float(iterations - iter) / float(iterations) * alpha_0

  V[N] = L_ter(x[:,N])
  Vx[:, N] = L_terx(x[:,N])
  Vxx[:, :, N] = L_terxx(x[:,N])

  for i in reversed(range(N)):
    fx_ = fx(x[:,i], u[:,i])
    fu_ = fu(x[:,i], u[:,i])

    Qx = (x_ter - x[:,i]).T @ Q + fx_.T @ Vx[:,i+1]
    Qu = u[:,i].T @ R + fu_.T @ Vx[:,i+1]

    Qxx = Q + fx_.T @ Vxx[:,:,i+1] @ fx_
    Quu = R + fu_.T @ Vxx[:,:,i+1] @ fu_
    Qux = fu_.T @ Vxx[:,:,i+1] @ fx_

    Quu_inv = np.linalg.inv(Quu)
    k[i] = - Quu_inv @ Qu
    K[i] = - Quu_inv @ Qux

    V[i] = V[i+1] - 0.5 * k[i].T @ Quu @ k[i]
    Vx[:, i] = np.array(Qx - K[i].T @ Quu @ k[i]).flatten()
    Vxx[:,:,i] = Qxx - K[i].T @ Quu @ K[i]

  xnew = np.zeros((n, N+1))
  xnew[:,0] = x[:,0]

  for i in range(N):
    u[:,i] = u[:,i] + alpha * k[i] + K[i] @ (xnew[:,i] - x[:,i])
    xnew[:,i+1] = np.array(f(xnew[:,i], u[:,i])).flatten()

  update_magnitude[iter] = np.linalg.norm(x-xnew)
  x = xnew

  elapsed_time = time.time() - start_time
  total_time += elapsed_time
  print('Iteration time: ', elapsed_time*1000, ' ms')

print('Total time: ', total_time*1000, ' ms')

xcheck = np.zeros((n, N+1))
xcheck[:,0] = np.array([0, 0, 0, 0])
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:,i], u[:,i])).flatten()

plt.figure()
plt.plot(update_magnitude)
plt.savefig('magn.png')
plt.close()

# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(N, x, u, p)


