import numpy as np
import math
import matplotlib.pyplot as plt
import casadi as cs
from matplotlib.animation import FuncAnimation

delta = 0.1
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
X = opt.variable(4)
U = opt.variable(1)

l, m1, m2, b1, b2 = (1, 10, 5, 0, 0)
f1 = lambda x, u: (l*m2*cs.sin(x[1])*x[3]**2 + u + m2*g*cs.cos(x[1])*cs.sin(x[1])) / (m1 + m2*(1-cs.cos(x[1])**2)) - b1*x[2]
f2 = lambda x, u: - (l*m2*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + (m1+m2)*g*cs.sin(x[1])) / (l*m1 + l*m2*(1-cs.cos(x[1])**2)) - b2*x[3]
f_ = lambda x, u: x + delta * cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
L_ = lambda x, u: x.T @ Q @ x + u.T @ R @ u
L_ter_ = lambda x_ter: 1000 * ((x_ter[1] - math.pi)**2 + x_ter[2]**2 + x_ter[3]**2)

f = cs.Function('f', [X, U], [f_(X,U)])
L = cs.Function('L', [X, U], [L_(X,U)])
L_ter = cs.Function('L_ter', [X], [L_ter_(X)])
fx = cs.Function('fx', [X, U], [cs.jacobian(f_(X,U), X)])
fu = cs.Function('fu', [X, U], [cs.jacobian(f_(X,U), U)])
Lx = cs.Function('Lx', [X, U], [cs.jacobian(L_(X,U), X)])
Lu = cs.Function('Lu', [X, U], [cs.jacobian(L_(X,U), U)])
Lxx = cs.Function('Lxx', [X, U], [cs.jacobian(cs.jacobian(L_(X,U), X).T, X)])
Lux = cs.Function('Lux', [X, U], [cs.jacobian(cs.jacobian(L_(X,U), U).T, X)])
Luu = cs.Function('Luu', [X, U], [cs.jacobian(cs.jacobian(L_(X,U), U).T, U)])
L_terx = cs.Function('L_terx', [X], [cs.jacobian(L_ter_(X), X)])
L_terxx = cs.Function('L_terxx', [X], [cs.jacobian(cs.jacobian(L_ter_(X), X), X)])

u = np.array([0.1 * np.ones(N)])

cost = 0
for i in range(N):
  x[:, i+1] = np.array(f(x[:,i], u[:,i])).flatten()
  cost = cost + L(x[:, i], u[:,i])

cost = cost + L_ter(x[:, N])

k = [np.array((m, 1))] * (N+1)
K = [np.array((m, n))] * (N+1)

for iter in range(iterations):
  V = np.zeros(N+1)
  V[N] = L_ter(x[:, N])
  Vx = np.zeros((n, N+1))
  Vx[:, N] = L_terx(x[:, N])
  Vxx = np.zeros((n, n, N+1))
  Vxx[:, :, N] = L_terxx(x[:, N])

  for i in reversed(range(N)):
    Qx = np.array(Lx(x[:, i], u[:,i])).flatten() + fx(x[:, i], u[:,i]).T @ Vx[:, i + 1]
    Qu = np.array(Lu(x[:, i], u[:,i])).flatten() + fu(x[:, i], u[:,i]).T @ Vx[:, i + 1]

    Qxx = Lxx(x[:, i], u[:,i]) + fx(x[:, i], u[:,i]).T @ Vxx[:, :, i + 1] @ fx(x[:, i], u[:,i])
    Qux = Lux(x[:, i], u[:,i]) + fu(x[:, i], u[:,i]).T @ Vxx[:, :, i + 1] @ fx(x[:, i], u[:,i])
    Quu = Luu(x[:, i], u[:,i]) + fu(x[:, i], u[:,i]).T @ Vxx[:, :, i + 1] @ fu(x[:, i], u[:,i])

    k[i] = -np.linalg.inv(Quu) @ Qu
    K[i] = -np.linalg.inv(Quu) @ Qux

    V[i] = V[i + 1] - 0.5 * k[i].T @ Quu @ k[i]
    Vx[:, i] = np.array(Qx - K[i].T @ Quu @ k[i]).flatten()
    Vxx[:, :, i] = Qxx - K[i].T @ Quu @ K[i]

  xnew = np.zeros((n, N+1))
  xnew[:, 0] = x[:, 0]
  unew = np.zeros((m, N))

  for i in range(N):
    unew[:, i] = u[:,i] + alpha * k[i] + K[i] @ (xnew[:, i] - x[:, i])
    xnew[:, i+1] = np.array(f(xnew[:, i], unew[:, i])).flatten()

  u = unew
  x = xnew

xcheck = np.zeros((n, N+1))
xcheck[:, 0] = np.array([0, 0, 0, 0])
for i in range(N):
  xcheck[:, i+1] = np.array(f(xcheck[:, i], u[:, i])).flatten()

def animate(i):
  np.set_printoptions(precision=2)

  plt.clf()
  plt.xlim((-5*l, 5*l))
  plt.ylim((-1.5*l, 1.5*l))
  plt.gca().set_aspect('equal')
    
  plt.plot(x[0,:][i] + np.array((l, l, - l, - l, + l))/4,  np.array((l, -l, -l, l, l))/4)
  plt.gca().add_patch(plt.Circle((x[0,:][i] + math.sin(x[1,:][i]), - math.cos(x[1,:][i])), l/8, color='blue'))
  plt.plot(np.array((x[0,:][i], x[0,:][i] + math.sin(x[1,:][i]))), np.array((0, - math.cos(x[1,:][i]))))

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)
plt.show()

