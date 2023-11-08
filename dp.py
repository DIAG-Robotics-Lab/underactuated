import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

def get_neighbors(matrix, row, col):
  neighbors = []
  directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  for dr, dc in directions:
    new_row, new_col = row + dr, col + dc
    if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
      neighbors.append(matrix[new_row, new_col])
  return neighbors

# Parameters
n, m = 10, 10
N = n+m
goal = 8, 8

# Matrix history
matrix = []
matrix.append(np.random.rand(n, m) * (n+m))
matrix[0][goal] = 0

# Run value iteration
for k in range(N):
  new_matrix = matrix[-1].copy()
  for i in range(n):
    for j in range(m):
      neighbors = get_neighbors(matrix[-1], i, j)
      if (i,j) == goal:
        new_matrix[i,j] = 0
      else:
        new_matrix[i,j] = min(neighbors) + 1
  matrix.append(new_matrix)

# Animate
def animate(frame_n):
  plt.clf()
  plt.gca().imshow(matrix[frame_n], cmap='gray', vmin=1, vmax=n+m)
  
anim = animation.FuncAnimation(plt.gcf(), animate, frames=N, repeat=False, interval=200, blit=False)
plt.show()


