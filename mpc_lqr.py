import numpy as np
import math
import scipy
import casadi as cs
import animation
import model
import time

# parameters
N = 20
N_sim = 100
delta_mpc = 0.01
delta_sim = 0.01
g = 9.81
u_max = 5000

# trajectory
u = np.zeros(N_sim)
x = np.zeros((4,N_sim+1))

# generate model
#f, F, p = model.get_cart_pendulum_model()
f, F, p = model.get_pendubot_model()

# optimization problem setup
opt = cs.Opti()
opt.solver("ipopt")

X = opt.variable(4,N+1)
U = opt.variable(1,N)
x0_param = opt.parameter(4)

# dynamics constraint
for i in range(N):
  opt.subject_to( X[:,i+1] == X[:,i] + delta_mpc * f(X[:,i], U[0,i]) )

# initial and terminal state
opt.subject_to( X[:,0] == x0_param )
x_term = np.array([[math.pi], [0], [0], [0]])
#opt.subject_to( X[(1,2,3),N] == (math.pi, 0, 0) )
#opt.subject_to( X[(0,1,2,3),N] == (math.pi, 0, 0, 0) )

# input constraint
#opt.subject_to( U <=   np.ones((1,N)) * u_max )
#opt.subject_to( U >= - np.ones((1,N)) * u_max )

# cost function
X_lqr = opt.variable(4)
U_lqr = opt.variable(1)

fx = cs.Function('fx', [X_lqr, U_lqr], [cs.jacobian(f(X_lqr,U_lqr), X_lqr)])
fu = cs.Function('fu', [X_lqr, U_lqr], [cs.jacobian(f(X_lqr,U_lqr), U_lqr)])

# solve LQR
A = np.array(fx([math.pi, 0, 0, 0], 0))
B = np.array(fu([math.pi, 0, 0, 0], 0))
R = np.identity(1)
Q = np.identity(4)
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

wu, wx0, wx1, wx2, wx3 = (1, 1, 1, 1, 1)
cost = wu*cs.sumsqr(U) + wx0*cs.sumsqr(X[0,:]) + wx1*cs.sumsqr(X[1,:]) + wx2*cs.sumsqr(X[2,:]) + wx3*cs.sumsqr(X[3,:]) + (X[:,N] - x_term).T @ P @ (X[:,N] - x_term)
opt.minimize(cost)

# iterate
elapsed_time = np.zeros(N_sim)
for j in range(N_sim):
  start_time = time.time()

  opt.set_value(x0_param, x[:,j])
  sol = opt.solve()
  u[j] = sol.value(U[:,0])
  
  # integrate
  x[:,j+1] = x[:,j] + delta_sim * F(x[:,j], u[j])

  elapsed_time[j] = time.time() - start_time

print('Total cost: ', cs.sumsqr(u))
print('Average computation time: ', np.mean(elapsed_time)*1000, ' ms')

# display
#animation.animate_cart_pendulum(N_sim, x, u, p)
animation.animate_pendubot(N_sim, x, u, p)
