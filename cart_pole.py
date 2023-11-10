import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

topt = cs.Opti()
topt.solver("ipopt")

# parameters
M = 10       # number of samples per shot
num = 20     # number of shooting points
N = M*num
delta = 0.01
g = 9.81
l, m1, m2, b1, b2 = (1, 10, 5, 0, 0)

# trajectory
u = np.zeros(N)
x = np.zeros((4,N+1))

# dynamics
f1 = lambda x, u: (l*m2*cs.sin(x[1])*x[3]**2 + u + m2*g*cs.cos(x[1])*cs.sin(x[1])) / (m1 + m2*(1-cs.cos(x[1])**2)) - b1*x[2]
f2 = lambda x, u: - (l*m2*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + (m1+m2)*g*cs.sin(x[1])) / (l*m1 + l*m2*(1-cs.cos(x[1])**2)) - b2*x[3]
f = lambda x, u: cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
F = lambda x, u: np.hstack( (x[2:4], f1(x, u), f2(x, u)) )

def direct_transcription():
  X = topt.variable(4,N+1)
  U = topt.variable(1,N)
  cost = 0

  for i in range(N):
    topt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[0,i]) )
    cost += U[0,i]**2

  topt.subject_to( X[:,0] == x[:, 0] )
  topt.subject_to( X[1,N] == math.pi )
  topt.subject_to( X[2:4,N] == (0, 0) )

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
  topt.subject_to( X[1,N] == math.pi )
  topt.subject_to( X[2:4,N] == (0, 0) )

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
  topt.subject_to( X[num-1][1,M] == math.pi )
  topt.subject_to( X[num-1][2:4,M] == (0, 0) )

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
  topt.subject_to( X[1,N] == math.pi )
  topt.subject_to( X[2:4,N] == (0, 0) )

  topt.minimize(cost)

  sol = topt.solve()
  return sol.value(U)


u = multiple_shooting()

# integrate
for i in range(N):
  x[:,i+1] = x[:,i] + delta * F(x[:,i], u[i])

# display
def animate(i):
  sz = l/4
  np.set_printoptions(precision=2)

  plt.clf()
  plt.xlim((-5*l, 5*l))
  plt.ylim((-1.5*l, 1.5*l))
  plt.gca().set_aspect('equal')
    
  plt.plot(x[0,:][i] + np.array((sz, sz, - sz, - sz, + sz)),  np.array((sz, -sz, -sz, sz, sz)))
  plt.gca().add_patch(plt.Circle((x[0,:][i] + math.sin(x[1,:][i]), - math.cos(x[1,:][i])), sz/2, color='blue'))
  plt.plot(np.array((x[0,:][i], x[0,:][i] + cs.sin(x[1,:][i]))), np.array((0, - cs.cos(x[1,:][i]))))

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)

plt.show()
