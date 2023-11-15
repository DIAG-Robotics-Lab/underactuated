import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
N = 20
N_sim = 200
delta = 0.01
g = 9.81

# trajectory
u = np.zeros(N_sim)
x = np.zeros((4,N_sim+1))
#x[1,0] = math.pi*2

# cart pole
l, m1, m2, b1, b2 = (1, 10, 5, 0, 0)
f1 = lambda x, u: (l*m2*cs.sin(x[1])*x[3]**2 + u + m2*g*cs.cos(x[1])*cs.sin(x[1])) / (m1 + m2*(1-cs.cos(x[1])**2)) - b1*x[2]
f2 = lambda x, u: - (l*m2*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + (m1+m2)*g*cs.sin(x[1])) / (l*m1 + l*m2*(1-cs.cos(x[1])**2)) - b2*x[3]
f = lambda x, u: cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
F = lambda x, u: np.hstack( (x[2:4], f1(x, u), f2(x, u)) )

# pendubot
"""m1, m2, I1, I2, l1, l2, d1, d2, fr1, fr2 = (1, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1)
a1, a2, a3, a4, a5 = (I1 + m1*d1**2 + I2 + m2*(l1**2+d2**2), m2*l1*d2, I2 + m2*d2**2, g * (m1*d1 + m2*l2), g*m2*d2)
m11 = lambda q, u: a1 + 2*a2*cs.cos(q[1])
m12 = lambda q, u: a3 + a2*cs.cos(q[1])
m22 = lambda q, u: a3
line1 = lambda q, u: - fr1*q[2] - a4*cs.sin(q[0]) - a5*cs.sin(q[0]+q[1]) - a2*cs.sin(q[1])*q[3]*(q[3]+2*q[2]) + u
line2 = lambda q, u: - fr2*q[3] - a5*cs.sin(q[0]+q[1]) - a2*cs.sin(q[1])*q[2]**2
f1 = lambda q, u: (  m22(q, u) * line1(q, u) - m12(q, u) * line2(q, u)) / (m11(q, u)*m22(q, u) - m12(q, u)**2)
f2 = lambda q, u: (- m12(q, u) * line1(q, u) + m11(q, u) * line2(q, u)) / (m11(q, u)*m22(q, u) - m12(q, u)**2)
f = lambda x, u: cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
F = lambda x, u: np.hstack( (x[2:4], f1(x, u), f2(x, u)) )"""

for j in range(N_sim):
  opt = cs.Opti()
  opt.solver("ipopt")

  X = opt.variable(4,N+1)
  U = opt.variable(1,N)
  
  cost = 0
  for i in range(N):
    # multiple shooting
    opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[0,i]) )
  
    # sum running cost
    cost += U[0,i]**2
  
  opt.subject_to( X[:,0] == x[:,j] )
  opt.subject_to( X[1,N] == math.pi )
  #opt.subject_to( X[0:2,N] == np.array((math.pi, 0)) )
  opt.subject_to( X[2:4,N] == np.array((0, 0)) )
  
  opt.minimize(cost)
  
  sol = opt.solve()
  u = sol.value(U)
  
  # integrate
  x[:,j+1] = x[:,j] + delta * F(x[:,j], u[0])

# display cart pendulum
def animate(i):
  np.set_printoptions(precision=2)

  plt.clf()
  plt.xlim((-5*l, 5*l))
  plt.ylim((-1.5*l, 1.5*l))
  plt.gca().set_aspect('equal')
    
  plt.plot(x[0,:][i] + np.array((l, l, - l, - l, + l))/4,  np.array((l, -l, -l, l, l))/4)
  plt.gca().add_patch(plt.Circle((x[0,i] + math.sin(x[1,i]), - math.cos(x[1,i])), l/8, color='blue'))
  plt.plot(np.array((x[0,i], x[0,i] + math.sin(x[1,i]))), np.array((0, - math.cos(x[1,i]))))
  
# display pendubot
"""def animate(i):
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
  plt.gca().add_patch(plt.Circle(p2, l1/10, color='blue'))"""

ani = FuncAnimation(plt.gcf(), animate, frames=N_sim, repeat=True, interval=delta*1000)

plt.show()
