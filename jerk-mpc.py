import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# parameters
N = 100
N_sim = 2000
delta = 0.01
g = 9.81
h = 0.75
eta = math.sqrt(g/h)

# trajectory
u = np.zeros((2,N_sim-N))
x = np.zeros((6,N_sim-N+1))

# lip model
A_lip = np.array([[0,1,0],[eta**2,0,-eta**2],[0,0,0]])
B_lip = np.array([[0],[0],[1]])

f = lambda x, u: cs.vertcat(
   A_lip @ x[:3] + B_lip @ u[0],
   A_lip @ x[3:] + B_lip @ u[1]
)

# bounds
zmp_x_max = np.array( [(i//100)*0.05 + 0.1 for i in range(N_sim)] )
zmp_x_min = np.array( [(i//100)*0.05 - 0.1 for i in range(N_sim)] )
zmp_y_max = np.array( [(((i%200)<100)-0.5)*0.1 + 0.1 for i in range(N_sim)] )
zmp_y_min = np.array( [(((i%200)<100)-0.5)*0.1 - 0.1 for i in range(N_sim)] )
zmp_x_mid = (zmp_x_min + zmp_x_max) / 2.0
zmp_y_mid = (zmp_y_min + zmp_y_max) / 2.0

opt = cs.Opti('conic')
opt.solver('osqp')

#opt = cs.Opti()
#opt.solver('ipopt')

U = opt.variable(2,N)
X = opt.variable(6,N+1)

cost = 0

x0_param = opt.parameter(6)
zmp_x_min_param = opt.parameter(N)
zmp_x_max_param = opt.parameter(N)
zmp_x_mid_param = opt.parameter(N)
zmp_y_min_param = opt.parameter(N)
zmp_y_max_param = opt.parameter(N)
zmp_y_mid_param = opt.parameter(N)


for i in range(N):
    opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[:,i]) )

cost = 0.1*cs.sumsqr(U[0,:]) + 0.1*cs.sumsqr(U[1,:]) + \
    cs.sumsqr(X[2, 1:].T - zmp_x_mid_param) + \
    cs.sumsqr(X[5, 1:].T - zmp_y_mid_param)

opt.subject_to( X[2,1:].T <= zmp_x_max_param )
opt.subject_to( X[2,1:].T >= zmp_x_min_param )
opt.subject_to( X[5,1:].T <= zmp_y_max_param )
opt.subject_to( X[5,1:].T >= zmp_y_min_param )

opt.subject_to( X[:,0] ==  x0_param)
opt.subject_to( X[0,N] == X[2,N] )
opt.subject_to( X[3,N] == X[5,N] )

opt.minimize(cost)

for j in range(N_sim-N):
  start_time = time.time()
  
  opt.set_value(x0_param, x[:,j])
  opt.set_value(zmp_x_min_param, zmp_x_min[j:j+N])
  opt.set_value(zmp_x_max_param, zmp_x_max[j:j+N])
  opt.set_value(zmp_x_mid_param, zmp_x_mid[j:j+N])
  opt.set_value(zmp_y_min_param, zmp_y_min[j:j+N])
  opt.set_value(zmp_y_max_param, zmp_y_max[j:j+N])
  opt.set_value(zmp_y_mid_param, zmp_y_mid[j:j+N])

  sol = opt.solve()
  
  u_ = sol.value(U)
  x_ = sol.value(X)

  u[:,j] = u_[:,0]
  x[:,j+1] = x_[:,1]
  
  end_time = time.time()
  elapsed_time = end_time - start_time

  print(f"My function took {elapsed_time} seconds to execute.")

# integrate
#for i in range(N):
#  x[:,i+1] = x[:,i] + delta * f(x[:,i], u[i,:])

# display
def animate(i):
  plt.clf()
  plt.axis((-0.1, 1, -0.5, 0.5))
  plt.gca().set_aspect('equal')
  
  plt.plot(np.array(x[2,0:i]), np.array(x[5,0:i]))
  plt.plot(np.array(x[0,0:i]), np.array(x[3,0:i]))

#ani = FuncAnimation(plt.gcf(), animate, frames=N_sim+1, repeat=False, interval=0)

plt.plot(np.array(x[2,:]), np.array(x[5,:]))
plt.plot(np.array(x[0,:]), np.array(x[3,:]))
plt.show()




