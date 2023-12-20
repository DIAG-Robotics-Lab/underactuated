import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import math

def animate_cart_pendulum(N_sim, x, u, p):
  grid = GridSpec(2, 2, width_ratios=[3, 1])
  ax_large = plt.subplot(grid[:, 0])
  ax_small1 = plt.subplot(grid[0, 1])
  ax_small2 = plt.subplot(grid[1, 1])

  x_max = max(x.min(), x.max(), key=abs)
  u_max = max(u.min(), u.max(), key=abs)

  def animate(i):
    ax_large.cla()
    ax_large.axis((-4, 4, -1.5, 1.5))
    ax_large.set_aspect('equal')
      
    ax_large.plot(x[0,:][i] + np.array((p.l, p.l, - p.l, - p.l, + p.l))/4,  np.array((p.l, -p.l, -p.l, p.l, p.l))/4)
    ax_large.add_patch(plt.Circle((x[0,:][i] + math.sin(x[1,:][i]), - math.cos(x[1,:][i])), p.l/8, color='blue'))
    ax_large.plot(np.array((x[0,:][i], x[0,:][i] + math.sin(x[1,:][i]))), np.array((0, - math.cos(x[1,:][i]))))

    ax_small1.cla()
    ax_small1.axis((0, N_sim, -x_max*1.1, x_max*1.1))
    ax_small1.plot(x[:,:i].T)

    ax_small2.cla()
    ax_small2.axis((0, N_sim, -u_max*1.1, u_max*1.1))
    ax_small2.plot(u[:i])

  ani = FuncAnimation(plt.gcf(), animate, frames=N_sim+1, repeat=True, interval=10)
  plt.show()

def animate_pendubot(N_sim, x, u, p, x_pred=None):
  grid = GridSpec(2, 2)
  ax_large = plt.subplot(grid[:, 0])
  ax_small1 = plt.subplot(grid[0, 1])
  ax_small2 = plt.subplot(grid[1, 1])

  x_max = max(x.min(), x.max(), key=abs)
  u_max = max(u.min(), u.max(), key=abs)

  def animate(i):
    ax_large.cla()
    ax_large.axis((-1.2, 1.2, -1.2, 1.2))
    ax_large.set_aspect('equal')

    p1 = (p.l1*math.sin(x[0,i]), -p.l1*math.cos(x[0,i]))
    p2 = (p.l1*math.sin(x[0,i]) + p.l2*math.sin(x[0,i]+x[1,i]), -p.l1*math.cos(x[0,i]) - p.l2*math.cos(x[0,i]+x[1,i]))

    if x_pred != None and i != 0:
      N_pred = x_pred[0].shape[1]
      p1_pred = np.zeros((2, N_pred))
      p2_pred = np.zeros((2, N_pred))
      for j in range(N_pred):
        p1_pred[:,j] = (p.l1*math.sin(x_pred[i-1][0,j]), -p.l1*math.cos(x_pred[i-1][0,j]))
        p2_pred[:,j] = (p.l1*math.sin(x_pred[i-1][0,j]) + p.l2*math.sin(x_pred[i-1][0,j]+x_pred[i-1][1,j]), \
                       -p.l1*math.cos(x_pred[i-1][0,j]) - p.l2*math.cos(x_pred[i-1][0,j]+x_pred[i-1][1,j]))
      ax_large.plot(p1_pred[0,:], p1_pred[1,:], color='orange')
      ax_large.plot(p2_pred[0,:], p2_pred[1,:], color='orange')
  
    ax_large.plot(np.array((0, p1[0])), np.array((0, p1[1])), color='blue')
    ax_large.plot(np.array((p1[0], p2[0])), np.array((p1[1], p2[1])), color='blue')
    ax_large.add_patch(plt.Circle(p1, p.l1/10, color='green'))
    ax_large.add_patch(plt.Circle(p2, p.l1/10, color='green'))

    ax_small1.cla()
    ax_small1.axis((0, N_sim, -x_max*1.1, x_max*1.1))
    ax_small1.plot(x[:,:i].T)

    ax_small2.cla()
    ax_small2.axis((0, N_sim, -u_max*1.1, u_max*1.1))
    ax_small2.plot(u[:i])

  ani = FuncAnimation(plt.gcf(), animate, frames=N_sim+1, repeat=True, interval=10)
  plt.show()