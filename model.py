import casadi as cs
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import math

class CartPendulum:
  def __init__(self):
    self.name = 'cart_pendulum'
    self.n, self.m = 4, 1
    self.g = 9.81
    self.Parameters = namedtuple('Parameters', ['l', 'm1', 'm2', 'b1', 'b2'])
    self.p = self.Parameters(l=1, m1=2, m2=1, b1=0, b2=0)
    self.f1 = lambda x, u: (self.p.l*self.p.m2*cs.sin(x[1])*x[3]**2 + u + self.p.m2*self.g*cs.cos(x[1])*cs.sin(x[1])) / (self.p.m1 + self.p.m2*(1-cs.cos(x[1])**2)) - self.p.b1*x[2]
    self.f2 = lambda x, u: - (self.p.l*self.p.m2*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + (self.p.m1+self.p.m2)*self.g*cs.sin(x[1])) / (self.p.l*self.p.m1 + self.p.l*self.p.m2*(1-cs.cos(x[1])**2)) - self.p.b2*x[3]
    self.f = lambda x, u: cs.vertcat( x[2:4], self.f1(x, u), self.f2(x, u) )

  def animate(self, N_sim, x, u):
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
        
      ax_large.plot(x[0,i] + np.array((self.p.l, self.p.l, - self.p.l, - self.p.l, + self.p.l))/4,  np.array((self.p.l, -self.p.l, -self.p.l, self.p.l, self.p.l))/4)
      ax_large.add_patch(plt.Circle((x[0,i] + math.sin(x[1,i]), - math.cos(x[1,i])), self.p.l/8, color='blue'))
      ax_large.plot(np.array((x[0,i], x[0,i] + math.sin(x[1,i]))), np.array((0, - math.cos(x[1,i]))))

      ax_small1.cla()
      ax_small1.axis((0, N_sim, -x_max*1.1, x_max*1.1))
      ax_small1.plot(x[:,:i].T)

      ax_small2.cla()
      ax_small2.axis((0, N_sim, -u_max*1.1, u_max*1.1))
      ax_small2.plot(u[:, :i].T)

    ani = FuncAnimation(plt.gcf(), animate, frames=N_sim+1, repeat=True, interval=10)
    plt.show()

class Pendubot:
  def __init__(self):
    self.name = 'pendubot'
    self.n, self.m = 4, 1
    self.g = 9.81
    self.Parameters = namedtuple('Parameters', ['m1', 'm2', 'I1', 'I2', 'l1', 'l2', 'd1', 'd2', 'fr1', 'fr2'])
    self.p = self.Parameters(m1=1, m2=1, I1=0, I2=0, l1=0.5, l2=0.5, d1=0.5, d2=0.5, fr1=0.1, fr2=0.1)
    self.a1, self.a2, self.a3, self.a4, self.a5 = (self.p.I1 + self.p.m1*self.p.d1**2 + self.p.I2 + self.p.m2*(self.p.l1**2+self.p.d2**2), self.p.m2*self.p.l1*self.p.d2, self.p.I2 + self.p.m2*self.p.d2**2, self.g * (self.p.m1*self.p.d1 + self.p.m2*self.p.l2), self.g*self.p.m2*self.p.d2)
    self.m11 = lambda q, u: self.a1 + 2*self.a2*cs.cos(q[1])
    self.m12 = lambda q, u: self.a3 + self.a2*cs.cos(q[1])
    self.m22 = lambda q, u: self.a3
    self.line1 = lambda q, u: - self.p.fr1*q[2] - self.a4*cs.sin(q[0]) - self.a5*cs.sin(q[0]+q[1]) - self.a2*cs.sin(q[1])*q[3]*(q[3]+2*q[2]) + u
    self.line2 = lambda q, u: - self.p.fr2*q[3] - self.a5*cs.sin(q[0]+q[1]) - self.a2*cs.sin(q[1])*q[2]**2
    self.f1 = lambda q, u: (  self.m22(q, u) * self.line1(q, u) - self.m12(q, u) * self.line2(q, u)) / (self.m11(q, u)*self.m22(q, u) - self.m12(q, u)**2)
    self.f2 = lambda q, u: (- self.m12(q, u) * self.line1(q, u) + self.m11(q, u) * self.line2(q, u)) / (self.m11(q, u)*self.m22(q, u) - self.m12(q, u)**2)
    self.f = lambda x, u: cs.vertcat( x[2:4], self.f1(x, u), self.f2(x, u) )

  def animate(self, N_sim, x, u):
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

      p1 = (self.p.l1*math.sin(x[0,i]), -self.p.l1*math.cos(x[0,i]))
      p2 = (self.p.l1*math.sin(x[0,i]) + self.p.l2*math.sin(x[0,i]+x[1,i]), -self.p.l1*math.cos(x[0,i]) - self.p.l2*math.cos(x[0,i]+x[1,i]))

      ax_large.plot(np.array((0, p1[0])), np.array((0, p1[1])), color='blue')
      ax_large.plot(np.array((p1[0], p2[0])), np.array((p1[1], p2[1])), color='blue')
      ax_large.add_patch(plt.Circle(p1, self.p.l1/10, color='green'))
      ax_large.add_patch(plt.Circle(p2, self.p.l1/10, color='green'))

      ax_small1.cla()
      ax_small1.axis((0, N_sim, -x_max*1.1, x_max*1.1))
      ax_small1.plot(x[:,:i].T)

      ax_small2.cla()
      ax_small2.axis((0, N_sim, -u_max*1.1, u_max*1.1))
      ax_small2.plot(u[:, :i].T)

    ani = FuncAnimation(plt.gcf(), animate, frames=N_sim+1, repeat=True, interval=10)
    plt.show()

class Uav:
  def __init__(self):
    self.name = 'uav'
    self.n, self.m = 6, 2
    self.g = 9.81
    self.Parameters = namedtuple('Parameters', ['m', 'I', 'fr_x', 'fr_z', 'fr_theta'])
    self.p = self.Parameters(m=1.0, I=0.01, fr_x=0.01, fr_z=0.01, fr_theta=0.01)
    self.f1 = lambda x, u: x[3]
    self.f2 = lambda x, u: x[4]
    self.f3 = lambda x, u: x[5]
    self.f4 = lambda x, u: (-self.p.fr_x * x[3] + u[0] * cs.sin(x[2])) / self.p.m
    self.f5 = lambda x, u: (-self.p.fr_z * x[4] - self.p.m * self.g + u[0] * cs.cos(x[2])) / self.p.m
    self.f6 = lambda x, u: (-self.p.fr_theta * x[5] + u[1]) / self.p.I
    self.f = lambda x, u: cs.vertcat(self.f1(x, u), self.f2(x, u), self.f3(x, u), self.f4(x, u), self.f5(x, u), self.f6(x, u))

  def animate(self, N_sim, x, u, x_pred=None):
    grid = GridSpec(2, 2)
    ax_large = plt.subplot(grid[:, 0])
    ax_small1 = plt.subplot(grid[0, 1])
    ax_small2 = plt.subplot(grid[1, 1])

    x_max = max(x.min(), x.max(), key=abs)
    u_max = max(u.min(), u.max(), key=abs)

    uav_length = 0.5

    def animate(i):
      ax_large.cla()
      ax_large.axis((-2, 2, -2, 2))
      ax_large.set_aspect('equal')

      x_pos = x[0, i]
      z_pos = x[1, i]
      theta = x[2, i]

      body_x1 = x_pos + uav_length * math.cos(theta) / 2
      body_z1 = z_pos - uav_length * math.sin(theta) / 2
      body_x2 = x_pos - uav_length * math.cos(theta) / 2
      body_z2 = z_pos + uav_length * math.sin(theta) / 2

      ax_large.plot([body_x1, body_x2], [body_z1, body_z2], color='blue', lw=2)
      ax_large.add_patch(plt.Circle((x_pos, z_pos), uav_length / 10, color='green'))

      if x_pred is not None and i != 0:
        N_pred = x_pred[0].shape[1]
        for j in range(N_pred):
          x_pos_pred = x_pred[i-1][0, j]
          z_pos_pred = x_pred[i-1][1, j]
          theta_pred = x_pred[i-1][2, j]

          body_x1_pred = x_pos_pred - uav_length * math.cos(theta_pred) / 2
          body_z1_pred = z_pos_pred - uav_length * math.sin(theta_pred) / 2
          body_x2_pred = x_pos_pred + uav_length * math.cos(theta_pred) / 2
          body_z2_pred = z_pos_pred + uav_length * math.sin(theta_pred) / 2

          ax_large.plot([body_x1_pred, body_x2_pred], [body_z1_pred, body_z2_pred], color='orange', lw=1)

      ax_small1.cla()
      ax_small1.axis((0, N_sim, -x_max * 1.1, x_max * 1.1))
      ax_small1.plot(x[:, :i].T)

      ax_small2.cla()
      ax_small2.axis((0, N_sim, -u_max * 1.1, u_max * 1.1))
      ax_small2.plot(u[:, :i].T)

    ani = FuncAnimation(plt.gcf(), animate, frames=N_sim + 1, repeat=True, interval=10)
    plt.show()