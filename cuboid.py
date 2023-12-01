import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

opt = cs.Opti()
opt.solver("ipopt")

# parameters
n_s = 6 # state: px, pz, theta, vx, vz, omega
n_i = 4 # input: f1x, f1z, f2x, f2z
N = 1000
delta = 0.01
g = 9.81
height = 1
mu = 1

# trajectory
u = np.zeros((n_i,N))
x = np.zeros((n_s,N+1))
x[:,0] = (0, 1, 0, 0, 0, 0)

# cart pole
m, w, I = (1, 1, 1)
f1 = lambda x, u: (u[0]+u[2]) / m
f2 = lambda x, u: (u[1]+u[3]) / m - g
f3 = lambda x, u: (x[1] * (u[0]+u[2]) / m + (-w/2-x[0])*u[1] + (w/2-x[0])*u[3]) / I
f = lambda x, u: cs.vertcat( x[3:6], f1(x, u), f2(x, u), f3(x, u) )
F = lambda x, u: np.hstack( (x[3:6], f1(x, u), f2(x, u), f3(x, u)) )

U = opt.variable(n_i,N)
X = opt.variable(n_s,N+1)

cost = 0
for i in range(N):
  # multiple shooting constraint
  opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[:,i]) )
  
  # unilaterality of contact forces
  opt.subject_to( U[1,i] >= 0 )
  opt.subject_to( U[3,i] >= 0 )

  # friction cones
  opt.subject_to( U[0,i] <=   mu * U[1,i] )
  opt.subject_to( U[0,i] >= - mu * U[1,i] )
  opt.subject_to( U[2,i] <=   mu * U[3,i] )
  opt.subject_to( U[2,i] >= - mu * U[3,i] )

  # zero angular momentum
  opt.subject_to( X[2,i] == 0 )
  
  # constant height
  opt.subject_to( X[1,i] == height )
  
  # balance
  cost += U[0,i]**2 + U[2,i]**2 + (U[1,i] - m*g/2)**2 + (U[3,i] - m*g/2)**2 + 100*X[2,i]**2

opt.subject_to( X[:,0] == x[:, 0] )
#opt.subject_to( X[:,int(N/2)] == np.array((1, 1, 0, 0, 0, 0)) )
opt.subject_to( X[:,int(N/2)] == np.array((0.4, 1, 0, 0, 0, 0)) )
opt.subject_to( X[:,N] == np.array((0, 1, 0, 0, 0, 0)) )

opt.minimize(cost)

sol = opt.solve()
u = sol.value(U)

# integrate
for i in range(N):
  x[:,i+1] = x[:,i] + delta * F(x[:,i], u[:,i])

# compute ZMP
x_zmp = np.zeros(N)
for i in range(N):
  x_zmp[i] = (- w/2 * u[1,i] + w/2 * u[3,i]) / (u[1,i] + u[3,i])

print(u)

# display cart pendulum
def animate(i):
  plt.clf()
  plt.xlim((-2, 2))
  plt.ylim((-0.5, 2.0))
  plt.gca().set_aspect('equal')
  
  force_scale = 0.2
   
  vertices = np.array([[-0.5, -0.5, 0.5, 0.5, -0.5], [-0.5, 0.5, 0.5, -0.5, -0.5]]) * w/2
  vertices = np.array([[math.cos(x[2,i]), - math.sin(x[2,i])], [math.sin(x[2,i]), math.cos(x[2,i])]]) @ vertices
  
  plt.plot(np.array((-w/2, w/2)), np.array((0, 0)), 'k')
  plt.plot(np.array((0, x[0,i])), np.array((0, x[1,i])), 'k')
  plt.plot(np.array((x[0,i], x[0,i])), np.array((0, x[1,i])), 'k--')
  plt.plot(x[0,i] + vertices[0,:], x[1,i] +  + vertices[1,:], 'k')
  
  plt.plot(np.array((- w/2, - w/2 + force_scale * u[0,i])), force_scale * np.array((0, u[1,i])), 'r')
  plt.plot(np.array((  w/2,   w/2 + force_scale * u[2,i])), force_scale * np.array((0, u[3,i])), 'r')

  #plt.plot(np.array((x_zmp[i], x[0,i])), np.array((0, x[1,i])), 'b')
  
ani = FuncAnimation(plt.gcf(), animate, frames=N, repeat=False, interval=0)

plt.show()
