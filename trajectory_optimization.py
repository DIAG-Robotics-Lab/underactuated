import numpy as np
import math
import casadi as cs
import animation
import model
import time

opt = cs.Opti()
opt.solver("ipopt")

# parameters
N = 200
delta = 0.01
g = 9.81

# trajectory
u = np.zeros(N)
x = np.zeros((4,N+1))

# generate model
#f, F, p = model.get_cart_pendulum_model()
f, F, p = model.get_pendubot_model()

# set up optimization problem
start_time = time.time()
X = opt.variable(4,N+1)
U = opt.variable(1,N+1)

for i in range(N):
  # multiple shooting
  opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[0,i]) )

  #single shooting
  #X[:,i+1] = X[:,i] + delta * f(X[:,i], U[0,i])

  #direct collocation
  #opt.subject_to( X[:,i+1] == X[:,i] + (delta/6) * (f(X[:,i], U[0,i]) + 4 * f((X[:,i]+X[:,i+1])/2 + (f(X[:,i], U[0,i])+f(X[:,i+1], U[0,i+1]))*delta/8, (U[0,i]+U[0,i+1])/2) + f(X[:,i+1], U[0,i+1])) )

opt.subject_to( X[:,0] == x[:, 0] )
#opt.subject_to( X[(1,2,3),N] == (math.pi, 0, 0) )
opt.subject_to( X[(0,1,2,3),N] == (math.pi, 0, 0, 0) )

wu, wx0, wx1, wx2, wx3 = (1, 0, 0, 0, 0)
cost = wu*cs.sumsqr(U) + wx0*cs.sumsqr(X[0,:]) + wx1*cs.sumsqr(X[1,:]) + wx2*cs.sumsqr(X[2,:]) + wx3*cs.sumsqr(X[3,:])
opt.minimize(cost)

sol = opt.solve()
u = sol.value(U)

# integrate
for i in range(N):
  x[:,i+1] = x[:,i] + delta * F(x[:,i], u[i])

elapsed_time = time.time() - start_time

print('Total cost: ', cs.sumsqr(u))
print('Average computation time: ', np.mean(elapsed_time)*1000, ' ms')

# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(N, x, u, p)