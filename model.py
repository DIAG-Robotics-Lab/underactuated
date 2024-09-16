import casadi as cs
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import math
import os

class BaseSystem:
    def __init__(self, name, n, m, g=9.81):
        self.name = name
        self.n = n
        self.m = m
        self.g = g

    def setup_animation(self, N_sim, x, u, save_frames):
        grid = GridSpec(2, 2)
        if not save_frames:
            self.ax_large = plt.subplot(grid[:, 0])
            self.ax_small1 = plt.subplot(grid[0, 1])
            self.ax_small2 = plt.subplot(grid[1, 1])
        else:
            self.ax_large = plt.subplot(grid[:, :])

        self.x_max = max(x.min(), x.max(), key=abs)
        self.u_max = max(u.min(), u.max(), key=abs)
        self.N_sim = N_sim

    def update_small_axes(self, x, u, i):
        self.ax_small1.cla()
        self.ax_small1.axis((0, self.N_sim, -self.x_max * 1.1, self.x_max * 1.1))
        self.ax_small1.plot(x[:, :i].T)

        self.ax_small2.cla()
        self.ax_small2.axis((0, self.N_sim, -self.u_max * 1.1, self.u_max * 1.1))
        self.ax_small2.plot(u[:, :i].T)

    def animate(self, N_sim, x, u, \
                show_trail=False, \
                save_video=False, video_filename="animation.mp4", \
                save_frames=False, frame_number=0, frame_folder="frames", \
                x_pred=None):

        self.setup_animation(N_sim, x, u, save_frames)
        x_pred.append(x_pred[-1]) # replicate last prediction to avoid crash

        frame_indices = np.linspace(0, self.N_sim, frame_number, dtype=int) if save_frames and frame_number > 0 else []

        if save_frames and frame_number > 0:
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

        def update_frame(i):
            self.ax_large.cla()

            trail_length = show_trail * 10
            spacing = 10
            trail_indices = [i - j * spacing for j in range(trail_length) if i - j * spacing >= 0]

            for idx, j in enumerate(trail_indices):
                alpha = 1.0 - (idx / (len(trail_indices) + 1))  # make older frames more faded
                alpha /= 4.  # make the trail more faded
                self.draw_frame(self.ax_large, j, x, u, alpha=alpha, x_pred=x_pred)

            self.draw_frame(self.ax_large, i, x, u, alpha=1., x_pred=x_pred)
            if not save_frames:
                self.update_small_axes(x, u, i)

            if save_frames and i in frame_indices:
                plt.savefig(os.path.join(frame_folder, f"frame_{i}.png"))

        ani = FuncAnimation(plt.gcf(), update_frame, frames=self.N_sim+1, repeat=True, interval=10)

        if save_video:
            writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(video_filename, writer=writer)
        else:
            plt.show()

class CartPendulum(BaseSystem):
    def __init__(self):
        super().__init__('cart_pendulum', 4, 1)
        self.Parameters = namedtuple('Parameters', ['l', 'm1', 'm2', 'b1', 'b2'])
        self.p = self.Parameters(l=1, m1=2, m2=1, b1=0, b2=0)
        self.f1 = lambda x, u: (self.p.l*self.p.m2*cs.sin(x[1])*x[3]**2 + u + self.p.m2*self.g*cs.cos(x[1])*cs.sin(x[1])) / (self.p.m1 + self.p.m2*(1-cs.cos(x[1])**2)) - self.p.b1*x[2]
        self.f2 = lambda x, u: - (self.p.l*self.p.m2*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + (self.p.m1+self.p.m2)*self.g*cs.sin(x[1])) / (self.p.l*self.p.m1 + self.p.l*self.p.m2*(1-cs.cos(x[1])**2)) - self.p.b2*x[3]
        self.f = lambda x, u: cs.vertcat( x[2:4], self.f1(x, u), self.f2(x, u) )

    def draw_frame(self, ax, i, x, u, alpha=1., x_pred=None):
        ax.axis((-1.5, 1.5, -1.5, 1.5))
        ax.set_aspect('equal')

        if x_pred is not None:
            x_p = x_pred[i]
            tip_x_pred = x_p[0, :] + np.sin(x_p[1, :])
            tip_y_pred = -np.cos(x_p[1, :])
            ax.plot(tip_x_pred, tip_y_pred, color='orange', alpha=alpha)

        ax.plot(x[0,i] + np.array((self.p.l, self.p.l, - self.p.l, - self.p.l, + self.p.l))/4,
                np.array((self.p.l, -self.p.l, -self.p.l, self.p.l, self.p.l))/4, color='orange', alpha=alpha)
        ax.add_patch(plt.Circle((x[0,i] + math.sin(x[1,i]), - math.cos(x[1,i])), self.p.l/8, color='blue', alpha=alpha))
        ax.plot(np.array((x[0,i], x[0,i] + math.sin(x[1,i]))), np.array((0, - math.cos(x[1,i]))), color='black', alpha=alpha)

class Pendubot(BaseSystem):
    def __init__(self):
        super().__init__('pendubot', 4, 1)
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

    def draw_frame(self, ax, i, x, u, alpha=1., x_pred=None):
        ax.axis((-1.2, 1.2, -1.2, 1.2))
        ax.set_aspect('equal')

        p1 = (self.p.l1 * math.sin(x[0,i]), -self.p.l1 * math.cos(x[0,i]))
        p2 = (self.p.l1 * math.sin(x[0,i]) + self.p.l2 * math.sin(x[0,i]+x[1,i]), -self.p.l1 * math.cos(x[0,i]) - self.p.l2 * math.cos(x[0,i]+x[1,i]))

        if x_pred is not None:
            x_p = x_pred[i]
            p1_pred_x = self.p.l1 * np.sin(x_p[0, :])
            p1_pred_y = -self.p.l1 * np.cos(x_p[0, :])
            p2_pred_x = self.p.l1 * np.sin(x_p[0, :]) + self.p.l2 * np.sin(x_p[0, :] + x_p[1, :])
            p2_pred_y = -self.p.l1 * np.cos(x_p[0, :]) - self.p.l2 * np.cos(x_p[0, :] + x_p[1, :])
            ax.plot(p1_pred_x, p1_pred_y, color='orange', alpha=alpha)
            ax.plot(p2_pred_x, p2_pred_y, color='orange', alpha=alpha)

        ax.plot(np.array((0, p1[0])), np.array((0, p1[1])), color='blue', alpha=alpha)
        ax.plot(np.array((p1[0], p2[0])), np.array((p1[1], p2[1])), color='blue', alpha=alpha)
        ax.add_patch(plt.Circle(p1, self.p.l1/10, color='green', alpha=alpha))
        ax.add_patch(plt.Circle(p2, self.p.l1/10, color='green', alpha=alpha))

class Uav(BaseSystem):
    def __init__(self):
        super().__init__('uav', 6, 2)
        self.Parameters = namedtuple('Parameters', ['m', 'I', 'fr_x', 'fr_z', 'fr_theta', 'width'])
        self.p = self.Parameters(m=1.0, I=0.01, fr_x=0.01, fr_z=0.01, fr_theta=0.01, width=0.2)

        self.f1 = lambda x, u: x[3]
        self.f2 = lambda x, u: x[4]
        self.f3 = lambda x, u: x[5]
        self.f4 = lambda x, u: (-0*self.p.fr_x * x[3] + (u[0] + u[1]) * cs.sin(x[2])) / self.p.m
        self.f5 = lambda x, u: (-0*self.p.fr_z * x[4] - self.p.m * self.g + (u[0] + u[1]) * cs.cos(x[2])) / self.p.m
        self.f6 = lambda x, u: (-0*self.p.fr_theta * x[5] + (self.p.width / 2) * (u[1] - u[0])) / self.p.I

        self.f = lambda x, u: cs.vertcat(self.f1(x, u), self.f2(x, u), self.f3(x, u), self.f4(x, u), self.f5(x, u), self.f6(x, u))

    def draw_frame(self, ax, i, x, u, alpha=1., x_pred=None):
        ax.axis((-2, 2, -2, 2))
        ax.set_aspect('equal')

        uav_length = 0.5
        uav_width = self.p.width
        x_pos = x[0, i]
        z_pos = x[1, i]
        theta = x[2, i]

        body_x1 = x_pos + (uav_length * math.cos(theta)) / 2 - (uav_width * math.sin(theta)) / 2
        body_z1 = z_pos - (uav_length * math.sin(theta)) / 2 - (uav_width * math.cos(theta)) / 2
        body_x2 = x_pos - (uav_length * math.cos(theta)) / 2 - (uav_width * math.sin(theta)) / 2
        body_z2 = z_pos + (uav_length * math.sin(theta)) / 2 - (uav_width * math.cos(theta)) / 2
        body_x3 = x_pos - (uav_length * math.cos(theta)) / 2 + (uav_width * math.sin(theta)) / 2
        body_z3 = z_pos + (uav_length * math.sin(theta)) / 2 + (uav_width * math.cos(theta)) / 2
        body_x4 = x_pos + (uav_length * math.cos(theta)) / 2 + (uav_width * math.sin(theta)) / 2
        body_z4 = z_pos - (uav_length * math.sin(theta)) / 2 + (uav_width * math.cos(theta)) / 2

        ax.plot([body_x1, body_x2, body_x3, body_x4, body_x1],
                [body_z1, body_z2, body_z3, body_z4, body_z1],
                color='blue', lw=2, alpha=alpha)

        ax.add_patch(plt.Circle((x_pos, z_pos), uav_length / 10, color='green', alpha=alpha))

        if x_pred is not None:
            x_p = x_pred[i]
            uav_x_pred = x_p[0, :]
            uav_y_pred = x_p[1, :]
            ax.plot(uav_x_pred, uav_y_pred, color='orange', alpha=alpha)

        thrust_scale = 0.05
        thrust_length_left  = max(u[1, i], 0.001) / self.p.m * thrust_scale
        thrust_length_right = max(u[0, i], 0.001) / self.p.m * thrust_scale

        left_x  = x_pos - (uav_width / 2) * math.cos(theta)
        left_z  = z_pos + (uav_width / 2) * math.sin(theta)
        right_x = x_pos + (uav_width / 2) * math.cos(theta)
        right_z = z_pos - (uav_width / 2) * math.sin(theta)

        thrust_x_left  = -thrust_length_left  * math.sin(theta)
        thrust_z_left  = -thrust_length_left  * math.cos(theta)
        thrust_x_right = -thrust_length_right * math.sin(theta)
        thrust_z_right = -thrust_length_right * math.cos(theta)

        ax.arrow(left_x , left_z , thrust_x_left , thrust_z_left , color='red', head_width=0.05, head_length=0.05, alpha=alpha)
        ax.arrow(right_x, right_z, thrust_x_right, thrust_z_right, color='red', head_width=0.05, head_length=0.05, alpha=alpha)