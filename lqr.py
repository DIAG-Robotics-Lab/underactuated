import numpy as np
import scipy
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

opt = cs.Opti()
opt.solver("ipopt")

# parameters
N = 200
delta = 0.01
g = 9.81

# trajectory
u = np.zeros(N)
x = np.zeros((4,N+1))
x[0,0] = math.pi + 0.2

# pendubot
m1, m2, I1, I2, l1, l2, d1, d2, fr1, fr2 = (1, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1)
a1, a2, a3, a4, a5 = (I1 + m1*d1**2 + I2 + m2*(l1**2+d2**2), m2*l1*d2, I2 + m2*d2**2, g * (m1*d1 + m2*l2), g*m2*d2)
m11 = lambda q, u: a1 + 2*a2*cs.cos(q[1])
m12 = lambda q, u: a3 + a2*cs.cos(q[1])
m22 = lambda q, u: a3
line1 = lambda q, u: - fr1*q[2] - a4*cs.sin(q[0]) - a5*cs.sin(q[0]+q[1]) - a2*cs.sin(q[1])*q[3]*(q[3]+2*q[2]) + u
line2 = lambda q, u: - fr2*q[3] - a5*cs.sin(q[0]+q[1]) - a2*cs.sin(q[1])*q[2]**2
f1 = lambda q, u: (  m22(q, u) * line1(q, u) - m12(q, u) * line2(q, u)) / (m11(q, u)*m22(q, u) - m12(q, u)**2)
f2 = lambda q, u: (- m12(q, u) * line1(q, u) + m11(q, u) * line2(q, u)) / (m11(q, u)*m22(q, u) - m12(q, u)**2)
f = lambda x, u: cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
F = lambda x, u: np.hstack( (x[2:4], f1(x, u), f2(x, u)) )

X = opt.variable(4)
U = opt.variable(1)

fx = cs.Function('fx', [X, U], [cs.jacobian(f(X,U), X)])
fu = cs.Function('fu', [X, U], [cs.jacobian(f(X,U), U)])

# solve LQR
A = np.array(fx([math.pi, 0, 0, 0], 0))
B = np.array(fu([math.pi, 0, 0, 0], 0))
R = np.identity(1)
Q = np.identity(4)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
print(P @ A + A.T @ P - P @ B @ np.linalg.inv(R) @ B.T @ P + Q)

# integrate
for i in range(N):
  u[i] = - np.linalg.inv(R) @ B.T @ P @ (x[:,i] - [math.pi, 0, 0, 0])
  x[:,i+1] = x[:,i] + delta * F(x[:,i], u[i])
  
# display pendubot
def animate(i):
  np.set_printoptions(precision=2)

  plt.clf()
  plt.xlim((-5*(l1+l2)), 5*(l1+l2))
  plt.ylim((-1.5*(l1+l2), 1.5*(l1+l2)))
  plt.gca().set_aspect('equal')
  
  p1 = (l1*math.sin(x[0,i]), -l1*math.cos(x[0,i]))
  p2 = (l1*math.sin(x[0,i]) + l2*math.sin(x[0,i]+x[1,i]), -l1*math.cos(x[0,i]) - l2*math.cos(x[0,i]+x[1,i]))
  
  plt.plot(np.array((0, p1[0])), np.array((0, p1[1])))
  plt.gca().add_patch(plt.Circle(p1, l1/10, color='blue'))
  plt.plot(np.array((p1[0], p2[0])), np.array((p1[1], p2[1])))
  plt.gca().add_patch(plt.Circle(p2, l1/10, color='blue'))

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)

plt.show()
