import numpy as np
import matplotlib.pyplot as plt

# Control points
P = np.array([[0, 0], [2, 5], [4, 2], [5, 5]])

# Number of points on the Bezier curve
n_points = 100

# Time interval
t_start = 0
t_end = 1

# Define the Bezier basis function
def b(i, n, t):
    return np.math.comb(n, i) * t**i * (1-t)**(n-i)

# Compute the trajectory points
t_values = np.linspace(t_start, t_end, n_points)
z_values = np.zeros((n_points, 2))
for i, t in enumerate(t_values):
    z_values[i] = sum([P[j]*b(j, len(P)-1, t) for j in range(len(P))])

# Plot the control points and trajectory
plt.plot(P[:,0], P[:,1], 'ro', label='Control Points')
plt.plot(z_values[:,0], z_values[:,1], 'b-', label='Trajectory')
plt.legend()
plt.show()
