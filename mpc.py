import numpy as np
import casadi as cs
import model
import time

# initialization
opt = cs.Opti()
p_opts, s_opts = {"ipopt.print_level": 0, "expand": True}, {}
opt.solver("ipopt", p_opts, s_opts)
mod = model.Pendubot()
N = 100
N_sim = 200
delta_mpc = 0.01
delta_sim = 0.01
u_max = 5000
f = mod.f

# trajectory
x = np.zeros((mod.n,N_sim+1))
u = np.zeros((mod.m,N_sim))

# optimization problem setup
X = opt.variable(mod.n,N+1)
U = opt.variable(mod.m,N)
x0_param = opt.parameter(mod.n)

for i in range(N):
  opt.subject_to( X[:,i+1] == X[:,i] + delta_mpc * f(X[:,i], U[:,i]) )

opt.subject_to( X[:,0] == x0_param )
if   mod.name == 'cart_pendulum': opt.subject_to( X[:,N] == (0, cs.pi, 0, 0) )
elif mod.name == 'pendubot'     : opt.subject_to( X[:,N] == (cs.pi, 0, 0, 0) )
elif mod.name == 'uav'          :
  opt.subject_to( X[:,N] == (1, 1, 0, 0, 0, 0) )
  for i in range(N):
    opt.subject_to( U[0,i] >= 0 )
    opt.subject_to( U[1,i] >= 0 )

# input constraint
#opt.subject_to( U <=   np.ones((mod.m,N)) * u_max )
#opt.subject_to( U >= - np.ones((mod.m,N)) * u_max )

# cost function
cost = cs.sumsqr(U)
opt.minimize(cost)

x_pred_record = []

# iterate
elapsed_time = np.zeros(N_sim)
for j in range(N_sim):
  start_time = time.time()

  # solve NLP
  opt.set_value(x0_param, x[:,j])
  sol = opt.solve()
  u[:,j] = sol.value(U[:,0])

  u_pred = sol.value(U)
  x_pred = sol.value(X)
  x_pred_record.append(x_pred)

  # set initial guess for next iteration
  opt.set_initial(U, u_pred)
  opt.set_initial(X, x_pred)
  
  # integrate
  x[:,j+1] = x[:,j] + delta_sim * f(x[:,j], u[:,j]).full().squeeze()

  elapsed_time[j] = time.time() - start_time

print('Average computation time: ', np.mean(elapsed_time) * 1000, ' ms')

# display
ani = mod.animate(N_sim, x, u, x_pred=x_pred_record)
#ani = mod.animate(N_sim, x, u, x_pred=x_pred_record, save_frames=True, frame_number=6)