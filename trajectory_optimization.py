import numpy as np
import casadi as cs
import model
import time

# initialization
opt = cs.Opti()
p_opts, s_opts = {"ipopt.print_level": 0, "expand": True}, {}
opt.solver("ipopt", p_opts, s_opts)
mod = model.Uav()
N = 200
delta = 0.01
f = mod.f

# trajectory
x = np.zeros((mod.n,N+1))
u = np.zeros((mod.m,N))

# optimization problem
start_time = time.time()
X = opt.variable(mod.n,N+1)
U = opt.variable(mod.m,N+1)

for i in range(N):
  opt.subject_to( X[:,i+1] == X[:,i] + delta * f(X[:,i], U[:,i]) )

opt.subject_to( X[:,0] == x[:, 0] )
if   mod.name == 'cart_pendulum': opt.subject_to( X[:,N] == (0, cs.pi, 0, 0) )
elif mod.name == 'pendubot'     : opt.subject_to( X[:,N] == (cs.pi, 0, 0, 0) )
elif mod.name == 'uav'          : 
  opt.subject_to( X[(0,1),N//2] == (-1, 1) )
  opt.subject_to( X[:,N]    == (1, 1, 0, 0, 0, 0) )

cost = cs.sumsqr(U)
opt.minimize(cost)

sol = opt.solve()
u = np.asmatrix(sol.value(U))

# integrate
for i in range(N):
  x[:,i+1] = x[:,i] + delta * f(x[:,i], u[:,i]).full().squeeze()

# results
elapsed_time = time.time() - start_time
print('Computation time: ', elapsed_time*1000, ' ms')
ani = mod.animate(N, x, u, save_frames=True, frame_number=6)