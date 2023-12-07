import numpy as np
import casadi as cs
from collections import namedtuple

def get_cart_pendulum_model():
  g = 9.81
  Parameters = namedtuple('Parameters', ['l', 'm1', 'm2', 'b1', 'b2'])
  #p = Parameters(l=1, m1=2, m2=1, b1=0, b2=0)
  p = Parameters(l=1, m1=10, m2=5, b1=0, b2=0)
  f1 = lambda x, u: (p.l*p.m2*cs.sin(x[1])*x[3]**2 + u + p.m2*g*cs.cos(x[1])*cs.sin(x[1])) / (p.m1 + p.m2*(1-cs.cos(x[1])**2)) - p.b1*x[2]
  f2 = lambda x, u: - (p.l*p.m2*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + (p.m1+p.m2)*g*cs.sin(x[1])) / (p.l*p.m1 + p.l*p.m2*(1-cs.cos(x[1])**2)) - p.b2*x[3]
  f = lambda x, u: cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
  F = lambda x, u: np.hstack( (x[2:4], f1(x, u), f2(x, u)) )

  return (f, F, p)

def get_pendubot_model():
  g = 9.81
  Parameters = namedtuple('Parameters', ['m1', 'm2', 'I1', 'I2', 'l1', 'l2', 'd1', 'd2', 'fr1', 'fr2'])
  p = Parameters(m1=1, m2=1, I1=0, I2=0, l1=0.5, l2=0.5, d1=0.5, d2=0.5, fr1=0.1, fr2=0.1)
  a1, a2, a3, a4, a5 = (p.I1 + p.m1*p.d1**2 + p.I2 + p.m2*(p.l1**2+p.d2**2), p.m2*p.l1*p.d2, p.I2 + p.m2*p.d2**2, g * (p.m1*p.d1 + p.m2*p.l2), g*p.m2*p.d2)
  m11 = lambda q, u: a1 + 2*a2*cs.cos(q[1])
  m12 = lambda q, u: a3 + a2*cs.cos(q[1])
  m22 = lambda q, u: a3
  line1 = lambda q, u: - p.fr1*q[2] - a4*cs.sin(q[0]) - a5*cs.sin(q[0]+q[1]) - a2*cs.sin(q[1])*q[3]*(q[3]+2*q[2]) + u
  line2 = lambda q, u: - p.fr2*q[3] - a5*cs.sin(q[0]+q[1]) - a2*cs.sin(q[1])*q[2]**2
  f1 = lambda q, u: (  m22(q, u) * line1(q, u) - m12(q, u) * line2(q, u)) / (m11(q, u)*m22(q, u) - m12(q, u)**2)
  f2 = lambda q, u: (- m12(q, u) * line1(q, u) + m11(q, u) * line2(q, u)) / (m11(q, u)*m22(q, u) - m12(q, u)**2)
  f = lambda x, u: cs.vertcat( x[2:4], f1(x, u), f2(x, u) )
  F = lambda x, u: np.hstack( (x[2:4], f1(x, u), f2(x, u)) )

  return (f, F, p)