import numpy as np
import math
import matplotlib.pyplot as plt
import casadi as cs
import time
import model
import animation
import random

class Node:
  def __init__(self, x, u, parent):
    self.children = []
    self.parent = parent
    self.x = x
    self.u = u

delta = 0.01
n, m = (4, 1)
iterations = 1000
goal_radius = 0.1
max_input = 100
max_state = 10
primitives = [-max_input, max_input]

x_ini = np.array([0, 0, 0, 0])
x_ter = np.array([math.pi, 0, 0, 0])

root = Node(x_ini, None, None)
tree = [root]

#f, p = model.get_cart_pendulum_model()
f, p = model.get_pendubot_model()

total_time = 0

for iter in range(iterations):
  start_time = time.time()

  # sample the state space
  sample = (np.random.rand(n) - 0.5) * 2 * max_state

  # find closest node
  distance = 10e6
  closest_node = None
  for node in tree:
    sample_distance = np.linalg.norm(sample - node.x)
    if sample_distance < distance:
      distance = sample_distance
      closest_node = node

  # expand
  random_input = primitives[random.randrange(len(primitives))]
  new_state = closest_node.x + delta * np.array(f(closest_node.x, random_input)).flatten()
  new_node = Node(new_state, random_input, closest_node)
  closest_node.children.append(new_node)
  tree.append(new_node)

  # check termination
  if np.linalg.norm(new_node.x - x_ter) < goal_radius:
    print('goal reached!')
    break

  elapsed_time = time.time() - start_time
  total_time += elapsed_time
  print('Iteration ', iter, ' time: ', elapsed_time*1000, ' ms')

# find best node
distance = 10e6
best_node = None
for node in tree:
  goal_distance = np.linalg.norm(x_ter - node.x)
  print(goal_distance)
  if goal_distance < distance:
    distance = goal_distance
    best_node = node

print('best ', best_node.x)

u_backwards = []
search_node = best_node
while search_node.parent != None:
  u_backwards.append(search_node.u)
  search_node = search_node.parent

u = np.flip(np.array(u_backwards))
x = np.zeros((n, len(u)+1))

for i in range(len(u)):
  x[:, i+1] = x[:,i] + delta * np.array(f(x[:,i], u[i])).flatten()

# display
#animation.animate_cart_pendulum(N, x, u, p)
animation.animate_pendubot(len(u), x, u, p)


