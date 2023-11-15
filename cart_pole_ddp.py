import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

delta = 0.01
N = 100
n, m = (4, 1)
Q = np.eye(n)
R = np.eye(m)
Qter = 100 * np.eye(n)
iterations = 100
alpha = 0.9
g = 9.81

u = np.array([0.1 * np.ones(N)], dtype='float32')
x = np.zeros((n, N+1), dtype='float32')
x[:, 0] = np.array([0, 0, 0, 0])

# dynamics and derivatives
x0, x1, x2, x3, u0 = sp.symbols('x0 x1 x2 x3 u0')
X = sp.Matrix([[x0], [x1], [x2], [x3]])
U = sp.Matrix([[u0]])

# cart pole
l, m1, m2, b1, b2 = (1, 10, 5, 0, 0)
f1_sym = (l*m2*sp.sin(X[1])*X[3]**2 + U[0] + m2*g*sp.cos(X[1])*sp.sin(X[1])) / (m1 + m2*(1-sp.cos(X[1])**2)) - b1*X[2]
f2_sym = - (l*m2*sp.cos(X[1])*sp.sin(X[1])*X[3]**2 + U[0]*sp.cos(X[1]) + (m1+m2)*g*sp.sin(X[1])) / (l*m1 + l*m2*(1-sp.cos(X[1])**2)) - b2*X[3]
f_sym = sp.Matrix([[X[2]], [X[3]], [f1_sym], [f2_sym]])

f = sp.lambdify([X, U], f_sym, 'numpy')
fx = sp.lambdify([X, U], f_sym.jacobian(X), 'numpy')
fu = sp.lambdify([X, U], f_sym.jacobian(U), 'numpy')

L_sym = X.T * Q * X + U.T * R * U
L = sp.lambdify([X, U], L_sym, 'numpy')
Lx = sp.lambdify([X, U], L_sym.jacobian(X), 'numpy')
Lu = sp.lambdify([X, U], L_sym.jacobian(U), 'numpy')
Lxx = sp.lambdify([X, U], L_sym.jacobian(X).jacobian(X), 'numpy')
Lux = sp.lambdify([X, U], L_sym.jacobian(U).jacobian(X), 'numpy')
Luu = sp.lambdify([X, U], L_sym.jacobian(U).jacobian(U), 'numpy')

#L_ter_sym = Qter * (X[1] - math.pi)**2
L_ter_sym = X.T * Qter * X
L_ter = sp.lambdify([X], L_ter_sym, 'numpy')
L_terx = sp.lambdify([X], L_ter_sym.jacobian(X), 'numpy')
L_terxx = sp.lambdify([X], L_ter_sym.jacobian(X).jacobian(X), 'numpy')

# integrate initial guess
cost = 0
for i in range(N):
  x[:, i+1] = f(x[:, i], u[:, i]).flatten()
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
    Qu = (Lu(x[:, i], u[:, i]) + fu(x[:, i], u[:, i]).T @ Vx[:, i + 1]).flatten()

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
    xnew[:, i+1] = f(xnew[:, i], unew[:, i]).flatten()

  u = unew
  x = xnew

  plt.plot(x[0, :], x[1, :], color=[0, 0, 1, iter / iterations])

xcheck = np.zeros((n, N+1))
xcheck[:, 0] = np.array([0, 0, 0, 0])
for i in range(N):
  xcheck[:, i+1] = f(xcheck[:, i], u[:, i]).flatten()

plt.plot(xcheck[0, :], xcheck[1, :], 'k')
plt.axis('equal')
plt.axis([-1.5, 1.5, -0.5, 1.5])

plt.show()

