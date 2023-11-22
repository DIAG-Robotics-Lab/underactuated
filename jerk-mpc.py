import numpy as np
import math
import casadi as cs
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# parameters
N = 100
N_sim = 1000
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

# moving constraint
zmp_x_mid = np.array( [(i//100)*0.05 for i in range(N_sim)] )
zmp_y_mid = np.array( [(((i%200)<100)-0.5)*0.1 for i in range(N_sim)] )

# optimization problem
opt = cs.Opti('conic')
opt.solver('osqp')

U = opt.variable(2,N)
X = opt.variable(6,N+1)

x0_param = opt.parameter(6)
zmp_x_mid_param = opt.parameter(N)
zmp_y_mid_param = opt.parameter(N)

for i in range(N):
  opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[:,i]) )

cost = 0.1*cs.sumsqr(U[0,:]) + 0.1*cs.sumsqr(U[1,:]) + \
    cs.sumsqr(X[2, 1:].T - zmp_x_mid_param) + \
    cs.sumsqr(X[5, 1:].T - zmp_y_mid_param)

opt.subject_to( X[2,1:].T <= zmp_x_mid_param + 0.1 )
opt.subject_to( X[2,1:].T >= zmp_x_mid_param - 0.1 )
opt.subject_to( X[5,1:].T <= zmp_y_mid_param + 0.1 )
opt.subject_to( X[5,1:].T >= zmp_y_mid_param - 0.1 )

opt.subject_to( X[:,0] ==  x0_param)
opt.subject_to( X[0,N] == X[2,N] )
opt.subject_to( X[3,N] == X[5,N] )

opt.minimize(cost)

elapsed_time = np.zeros(N_sim-N)

for j in range(N_sim-N):
  start_time = time.time()

  opt.set_value(x0_param, x[:,j])
  opt.set_value(zmp_x_mid_param, zmp_x_mid[j:j+N])
  opt.set_value(zmp_y_mid_param, zmp_y_mid[j:j+N])
  
  sol = opt.solve()

  u[:,j] = sol.value(U[:,0])
  x[:,j+1] = sol.value(X[:,1])

  end_time = time.time()
  
  elapsed_time[j] = end_time - start_time

print(f"Average computation time: {np.mean(elapsed_time)*1000} ms.")

# display
def animate(i):
  plt.clf()
  plt.axis((-0.1, 1, -0.5, 0.5))
  plt.gca().set_aspect('equal')
  
  plt.plot(np.array(x[2,0:i]), np.array(x[5,0:i]))
  plt.plot(np.array(x[0,0:i]), np.array(x[3,0:i]))

ani = FuncAnimation(plt.gcf(), animate, frames=N_sim+1, repeat=False, interval=0)

plt.show()




