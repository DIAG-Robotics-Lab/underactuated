import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#opt = cs.Opti()
#opt.solver('ipopt')
opt = cs.Opti('conic')
opt.solver('osqp')

# parameters
N = 1000
delta = 0.01
g = 9.81
h = 0.75
eta = math.sqrt(g/h)

# trajectory
#u = np.zeros((2,N))
#x = np.zeros((6,N+1))

# lip model
A_lip = np.array([[0,1,0],[eta**2,0,-eta**2],[0,0,0]])
B_lip = np.array([[0],[0],[1]])
f = lambda x, u: (np.vstack((np.hstack((A_lip, np.zeros((3,3)))), np.hstack((np.zeros((3,3)), A_lip)))) @ x +
                  np.vstack((np.hstack((B_lip, np.zeros((3,1)))), np.hstack((np.zeros((3,1)), B_lip)))) @ u)

# bounds
zmp_x_mid = np.array( [(i//100)*0.1 for i in range(N)] )
zmp_y_mid = np.array( [(((i%200)<100)-0.5)*0.1 for i in range(N)] )

zmp_x_max = np.array( [(i//100)*0.1 + 0.1 for i in range(N)] )
zmp_x_min = np.array( [(i//100)*0.1 - 0.1 for i in range(N)] )
zmp_y_max = np.array( [(((i%200)<100)-0.5)*0.1 + 0.1 for i in range(N)] )
zmp_y_min = np.array( [(((i%200)<100)-0.5)*0.1 - 0.1 for i in range(N)] )

U = opt.variable(2,N)
X = opt.variable(6,N+1)

cost = 0
for i in range(N):
  opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[:,i]) )
  #X[:,i+1] == X[:,i] + delta * f(X[:,i], U[:,i])
  #cost += U[0,i]**2 + U[1,i]**2 + (X[2,i]-zmp_x_mid[i])**2 + (X[5,i]-zmp_y_mid[i])**2
  cost += (X[2,i]-zmp_x_mid[i])**2 + (X[5,i]-zmp_y_mid[i])**2
  
  opt.subject_to( X[2,i+1] <= zmp_x_max[i] )
  opt.subject_to( X[2,i+1] >= zmp_x_min[i] )
  opt.subject_to( X[5,i+1] <= zmp_y_max[i] )
  opt.subject_to( X[5,i+1] >= zmp_y_min[i] )

opt.subject_to( X[:,0] == (0,0,0,0,0,0) )
opt.subject_to( X[0,N] == X[2,N] )
opt.subject_to( X[3,N] == X[5,N] )

opt.minimize(cost)

sol = opt.solve()
u = sol.value(U)
x = sol.value(X)

# integrate
#for i in range(N):
#  x[:,i+1] = x[:,i] + delta * f(x[:,i], u[i,:])

# display
def animate(i):
  np.set_printoptions(precision=2)

  plt.clf()
  plt.xlim((-0.1, 1))
  plt.ylim((-0.5, 0.5))
  plt.gca().set_aspect('equal')
  
  plt.plot(np.array(x[2,0:i]), np.array(x[5,0:i]))
  plt.plot(np.array(x[0,0:i]), np.array(x[3,0:i]))

ani = FuncAnimation(plt.gcf(), animate, frames=N+1, repeat=False, interval=0)

plt.show()
