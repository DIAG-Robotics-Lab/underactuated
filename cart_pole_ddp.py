import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from sympy import *

delta = 0.1
N = 100
Q = np.eye(3)
R = np.eye(2)
Qter = 100 * np.eye(3)
iterations = 100
n = 3
m = 2
alpha = 0.9

x = np.zeros((n, N+1))
x[:, 0] = np.array([0, 1, 0])

topt = Opti()
X1 = topt.variable(1)
X2 = topt.variable(1)
X3 = topt.variable(1)
X = np.array((X1, X2, X3))
U = np.array((topt.variable(1)))

f = lambda x, u: x + delta * np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])
L = lambda x, u: x.T @ Q @ x + u.T @ R @ u
L_ter = lambda x_ter: x_ter.T @ Qter @ x_ter

XX = np.array((symbols('X1'), symbols('X2'), symbols('X3')))
UU = np.array((symbols('UU')))
print(f(XX, UU).diff(XX))

FX = jacobian(f(X,U),X)
FU = jacobian(f(X,U),U)
LX = jacobian(L(X,U),X)
LU = jacobian(L(X,U),U)
LXX = jacobian(LX,X)
LUX = jacobian(LU,X)
LUU = jacobian(LU,U)
L_TERX = jacobian(L_ter(X,U),X)
L_TERXX = jacobian(L_TERX,X)

fu = lambda x, u: FX(x, u)
fu = lambda x, u: FU(x, u)
Lx = lambda x, u: LX(x, u)
Lu = lambda x, u: FU(x, u)
Lxx = lambda x, u: LXX(x, u)
Lux = lambda x, u: LUX(x, u)
Luu = lambda x, u: LUU(x, u)
L_terx = lambda x_ter: L_TERX(x_ter)
L_terxx = lambda x_ter: L_TERXX(x_ter)

"""fx = lambda x, u: np.eye(n) + delta * np.array([[0, 0, -u[0] * np.sin(x[2])], [0, 0, u[0] * np.cos(x[2])], [0, 0, 0]])
fu = lambda x, u: delta * np.array([[np.cos(x[2]), 0], [np.sin(x[2]), 0], [0, 1]])
Lx = lambda x, u: 2 * Q @ x
Lu = lambda x, u: 2 * R @ u
Lxx = lambda x, u: 2 * Q
Lux = lambda x, u: np.zeros((m, n))
Luu = lambda x, u: 2 * R
L_terx = lambda x_ter: 2 * Qter @ x_ter
L_terxx = lambda x_ter: 2 * Qter"""

u = np.array([0.15 * np.ones(N), -0.3 * np.ones(N)])

cost = 0
for i in range(N):
  x[:, i+1] = f(x[:, i], u[:, i])
  cost = cost + L(x[:, i], u[:, i])

cost = cost + L_ter(x[:, N])

k = [np.array((m, 1))] * (N+1)
K = [np.array((m, n))] * (N+1)

plt.figure()

for iter in range(iterations):
  V = np.zeros(N+1)
  V[N] = L_ter(x[:, N])
  Vx = np.zeros((n, N+1))
  Vx[:, N] = L_terx(x[:, N])
  Vxx = np.zeros((n, n, N+1))
  Vxx[:, :, N] = L_terxx(x[:, N])

  for i in reversed(range(N)):
    Qx = Lx(x[:, i], u[:, i]) + fx(x[:, i], u[:, i]).T @ Vx[:, i + 1]
    Qu = Lu(x[:, i], u[:, i]) + fu(x[:, i], u[:, i]).T @ Vx[:, i + 1]

    Qxx = Lxx(x[:, i], u[:, i]) + fx(x[:, i], u[:, i]).T @ Vxx[:, :, i + 1] @ fx(x[:, i], u[:, i])
    Qux = Lux(x[:, i], u[:, i]) + fu(x[:, i], u[:, i]).T @ Vxx[:, :, i + 1] @ fx(x[:, i], u[:, i])
    Quu = Luu(x[:, i], u[:, i]) + fu(x[:, i], u[:, i]).T @ Vxx[:, :, i + 1] @ fu(x[:, i], u[:, i])

    k[i] = -np.linalg.inv(Quu) @ Qu
    K[i] = -np.linalg.inv(Quu) @ Qux

    V[i] = V[i + 1] - 0.5 * k[i].T @ Quu @ k[i]
    Vx[:, i] = Qx - K[i].T @ Quu @ k[i]
    Vxx[:, :, i] = Qxx - K[i].T @ Quu @ K[i]

  xnew = np.zeros((n, N+1))
  xnew[:, 0] = x[:, 0]
  unew = np.zeros((m, N))

  for i in range(N):
    unew[:, i] = u[:, i] + alpha * k[i] + K[i] @ (xnew[:, i] - x[:, i])
    xnew[:, i+1] = f(xnew[:, i], unew[:, i])

  u = unew
  x = xnew

  plt.plot(x[0, :], x[1, :], color=[0, 0, 1, iter / iterations])

xcheck = np.zeros((n, N+1))
xcheck[:, 0] = np.array([0, 1, 0])
for i in range(N):
  xcheck[:, i+1] = f(xcheck[:, i], u[:, i])

plt.plot(xcheck[0, :], xcheck[1, :], 'k')
plt.axis('equal')
plt.axis([-1.5, 1.5, -0.5, 1.5])

plt.show()

