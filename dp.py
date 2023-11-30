import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

def get_neighbors(matrix, row, col):
  neighbors = []
  directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]# + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
  for dr, dc in directions:
    new_row, new_col = row + dr, col + dc
    if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
      neighbors.append(matrix[new_row, new_col])
  return neighbors

# Parameters
n, m = 10, 10
N = n+m
goal = 8, 8
obstacles = [(2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9)]
obsacle_cost_to_go = 100

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
      elif (i,j) in obstacles:
        new_matrix[i,j] = obsacle_cost_to_go
      else:
        new_matrix[i,j] = min(neighbors) + 1
  matrix.append(new_matrix)

# Animate
def animate(frame_n):
  plt.clf()
  plt.gca().imshow(matrix[frame_n], cmap='gray_r', vmin=1, vmax=n+m)
  rect = patches.Rectangle((goal[1]-0.5, goal[0]-0.5), 1, 1, linewidth=2, edgecolor='green', facecolor='none')
  plt.gca().add_patch(rect)
  for i in range(10):
    for j in range(10):
      plt.text(i, j, str(int(matrix[frame_n][j, i])), color='blue', ha='center', va='center', fontsize=10)
  for obs in obstacles:
    rect = patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
  
anim = animation.FuncAnimation(plt.gcf(), animate, frames=N, repeat=False, interval=200, blit=False)
plt.show()


