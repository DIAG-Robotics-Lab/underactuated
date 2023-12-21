import numpy as np
import math
import casadi as cs
import animation
import model
import time

# parameters
N = 100
N_sim = 200
delta_mpc = 0.01
delta_sim = 0.01
g = 9.81
u_max = 5000

# trajectory
u = np.zeros(N_sim)
x = np.zeros((4,N_sim+1))

# generate model
#f, p = model.get_cart_pendulum_model()
f, p = model.get_pendubot_model()

# optimization problem setup
opt = cs.Opti()
p_opts = {"ipopt.print_level": 0, "expand": True}
s_opts = {} #{"max_iter": 1}
opt.solver("ipopt", p_opts, s_opts)

X = opt.variable(4,N+1)
U = opt.variable(1,N)
x0_param = opt.parameter(4)

# dynamics constraint
for i in range(N):
  opt.subject_to( X[:,i+1] == X[:,i] + delta_mpc * f(X[:,i], U[0,i]) )

# initial and terminal state
x_ter = np.array((math.pi, 0, 0, 0))
opt.subject_to( X[:,0] == x0_param )
#opt.subject_to( X[(1,2,3),N] == (math.pi, 0, 0) )
opt.subject_to( X[(0,1,2,3),N] == x_ter )

# input constraint
#opt.subject_to( U <=   np.ones((1,N)) * u_max )
#opt.subject_to( U >= - np.ones((1,N)) * u_max )

# cost function
wu, wx0, wx1, wx2, wx3 = (1, 1, 1, 1, 1)
cost = wu*cs.sumsqr(U) + wx0*cs.sumsqr(X[0,:]-x_ter[0]) + wx1*cs.sumsqr(X[1,:]-x_ter[1]) + wx2*cs.sumsqr(X[2,:]-x_ter[2]) + wx3*cs.sumsqr(X[3,:]-x_ter[3])
opt.minimize(cost)

x_pred_record = []

# iterate
elapsed_time = np.zeros(N_sim)
for j in range(N_sim):
  start_time = time.time()

  # solve NLP
  opt.set_value(x0_param, x[:,j])
  sol = opt.solve()
  u[j] = sol.value(U[:,0])

  u_pred = sol.value(U)
  x_pred = sol.value(X)
  x_pred_record.append(x_pred)

  # set initial guess for next iteration
  opt.set_initial(U, u_pred)
  opt.set_initial(X, x_pred)
  
  # integrate
  x[:,j+1] = x[:,j] + delta_sim * f(x[:,j], u[j]).full().squeeze()

  elapsed_time[j] = time.time() - start_time

print('Total cost: ', cs.sumsqr(u))
print('Average computation time: ', np.mean(elapsed_time)*1000, ' ms')

# display
#animation.animate_cart_pendulum(N_sim, x, u, p)
animation.animate_pendubot(N_sim, x, u, p, x_pred_record)
