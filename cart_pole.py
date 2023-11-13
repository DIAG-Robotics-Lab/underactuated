import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

topt = cs.Opti()
topt.solver("ipopt")

# parameters
M = 20       # number of samples per shot
num = 20     # number of shooting points
N = M*num
delta = 0.01
g = 9.81

# trajectory
u = np.zeros(N)
x = np.zeros((4,N+1))
#x[0,0] = math.pi*0.6

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

# terminal state
#term_pos = np.array((math.pi, 0))
term_pos = np.array(math.pi)
term_vel = np.array((0, 0))

def direct_transcription():
  X = topt.variable(4,N+1)
  U = topt.variable(1,N)
  cost = 0

  for i in range(N):
    topt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[0,i]) )
    cost += U[0,i]**2

  topt.subject_to( X[:,0] == x[:, 0] )
  topt.subject_to( X[1,N] == math.pi )
  #topt.subject_to( X[0:2,N] == term_pos )
  topt.subject_to( X[2:4,N] == term_vel )

  topt.minimize(cost)

  sol = topt.solve()
  return sol.value(U)
  
def single_shooting():
  X = topt.variable(4,N+1)
  U = topt.variable(1,N)
  cost = 0

  for i in range(N):
    X[:,i+1] = X[:,i] + delta * f(X[:,i], U[0,i])
    cost += U[0,i]**2

  topt.subject_to( X[:,0] == x[:, 0] )
  #topt.subject_to( X[1,N] == math.pi )
  topt.subject_to( X[0:2,N] == term_pos )
  topt.subject_to( X[2:4,N] == term_vel )

  topt.minimize(cost)

  sol = topt.solve()
  return sol.value(U)
  
def multiple_shooting():
  X = []
  U = topt.variable(1,N)
  cost = 0

  for j in range(num):
    X.append(topt.variable(4,M+1))
    for i in range(M):
      X[j][:,i+1] = X[j][:,i] + delta * f(X[j][:,i], U[0,j*M+i])
        
  for j in range(num-1):
    topt.subject_to( X[j][:,M] == X[j+1][:,0] )
        
  for i in range(N):
    cost += U[0,i]**2

  topt.subject_to( X[0][:,0] == x[:, 0] )
  #topt.subject_to( X[num-1][1,M] == math.pi )
  topt.subject_to( X[num-1][0:2,M] == term_pos )
  topt.subject_to( X[num-1][2:4,M] == term_vel )

  topt.minimize(cost)

  sol = topt.solve()
  return sol.value(U)
  
def direct_collocation():
  X = topt.variable(4,N+1)
  U = topt.variable(1,N+1)
  cost = 0

  for i in range(N):
    topt.subject_to( X[:,i+1] == X[:,i] + (delta/6) * (f(X[:,i], U[0,i]) + 4 * f((X[:,i]+X[:,i+1])/2 + (f(X[:,i], U[0,i])+f(X[:,i+1], U[0,i+1]))*delta/8, (U[0,i]+U[0,i+1])/2) + f(X[:,i+1], U[0,i+1])) )
    cost += U[0,i]**2

  topt.subject_to( X[:,0] == x[:, 0] )
  #topt.subject_to( X[1,N] == math.pi )
  topt.subject_to( X[0:2,N] == term_pos )
  topt.subject_to( X[2:4,N] == term_vel )

  topt.minimize(cost)

  sol = topt.solve()
  return sol.value(U)

u = direct_transcription()

# integrate
for i in range(N):
  x[:,i+1] = x[:,i] + delta * F(x[:,i], u[i])

# display cart pendulum
def animate(i):
  np.set_printoptions(precision=2)

  plt.clf()
  plt.xlim((-5*l, 5*l))
  plt.ylim((-1.5*l, 1.5*l))
  plt.gca().set_aspect('equal')
    
  plt.plot(x[0,:][i] + np.array((l, l, - l, - l, + l))/4,  np.array((l, -l, -l, l, l))/4)
  plt.gca().add_patch(plt.Circle((x[0,:][i] + math.sin(x[1,:][i]), - math.cos(x[1,:][i])), l/8, color='blue'))
  plt.plot(np.array((x[0,:][i], x[0,:][i] + math.sin(x[1,:][i]))), np.array((0, - math.cos(x[1,:][i]))))
  
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

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)

plt.show()
