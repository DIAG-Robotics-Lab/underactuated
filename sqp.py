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
Qter = 1000 * np.eye(n)
iterations = 100
g = 9.81

alpha = 0.9

x = np.zeros((n, N+1))
x[:, 0] = np.array([0, 0, 0, 0])

opt = cs.Opti()
X_ = opt.variable(4)
U_ = opt.variable(1)

L_ = lambda x, u: x.T @ Q @ x + u.T @ R @ u
#L_ter_ = lambda x_ter: 1000 * ((x_ter[1] - math.pi)**2 + x_ter[2]**2 + x_ter[3]**2)
L_ter_ = lambda x_ter: 10000 * ((x_ter[0] - math.pi)**2 + x_ter[1]**2 + x_ter[2]**2 + x_ter[3]**2)

#ff, F, p = model.get_cart_pendulum_model()
ff, F, p = model.get_pendubot_model()
f_ = lambda x, u: x + delta * ff(x, u)

f = cs.Function('f', [X_, U_], [f_(X_,U_)], {"post_expand": True})
L = cs.Function('L', [X_, U_], [L_(X_,U_)], {"post_expand": True})
L_ter = cs.Function('L_ter', [X_], [L_ter_(X_)], {"post_expand": True})
fx = cs.Function('fx', [X_, U_], [cs.jacobian(f_(X_,U_), X_)], {"post_expand": True})
fu = cs.Function('fu', [X_, U_], [cs.jacobian(f_(X_,U_), U_)], {"post_expand": True})
L_terx = cs.Function('L_terx', [X_], [cs.jacobian(L_ter_(X_), X_)], {"post_expand": True})
L_terxx = cs.Function('L_terxx', [X_], [cs.jacobian(cs.jacobian(L_ter_(X_), X_), X_)], {"post_expand": True})

u = np.ones(N) * 10

cost = 0
for i in range(N):
  x[:, i+1] = np.array(f(x[:,i], u[i])).flatten()
  cost = cost + L(x[:,i], u[i])

cost = cost + L_ter(x[:, N])

xnew = np.zeros((n, N+1))

total_time = 0
update_magnitude = np.zeros(iterations)

# optimization problem
opt = cs.Opti('conic')
opt.solver('proxqp')
#opt = cs.Opti()
#p_opts = {"ipopt.print_level": 0, "expand": True}
#s_opts = {} #{"max_iter": 1000}
#opt.solver("ipopt", p_opts, s_opts)

X = opt.variable(4,N+1)
U = opt.variable(1,N+1)

A = [opt.parameter(n,n)] * N
B = [opt.parameter(n,m)] * N

opt.subject_to( X[:,0] == x[:, 0] )
opt.subject_to( X[:,N] == [math.pi, 0, 0, 0] )
for i in range(N):
  opt.subject_to( X[:,i+1] == A[i] @ X[:,i] + B[i] @ U[:,i] )

cost = 0
for i in range(N):
  cost = cost + L(X[:,i], U[:,i])
cost = cost + L_ter(X[:, N])

opt.minimize(cost)

for iter in range(iterations):
  start_time = time.time()

  for i in range(N):
    opt.set_value(A[i], fx(x[:,i], u[i]))
    opt.set_value(B[i], fu(x[:,i], u[i]))

  sol = opt.solve()
  u = sol.value(U)
  xnew = sol.value(X)

  update_magnitude[iter] = np.linalg.norm(x-xnew)
  x = xnew

  elapsed_time = time.time() - start_time
  total_time += elapsed_time
  print('Iteration time: ', elapsed_time*1000, ' ms')

print('Total time: ', total_time*1000, ' ms')

xcheck = np.zeros((n, N+1))
xcheck[:,0] = np.array([0, 0, 0, 0])
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:,i], u[i])).flatten()

plt.figure()
plt.plot(update_magnitude)
plt.savefig('magn.png')
plt.close()

# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(N, x, u, p)


