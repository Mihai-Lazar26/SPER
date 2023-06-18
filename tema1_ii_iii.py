import numpy as np
import casadi
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt

### ii

# Define the number of control points
NUM_CONTROL_POINTS = 5

# Define the number of intermediary points / time moments
NUM_INTERMEDIARY_POINTS = 5

# Define the intermediary points and time moments
intermediary_points = np.array([[0, 0], [2, 10], [5, -10], [7, 10], [10, 10]])
time_moments = np.array([0, 0.25, 0.50, 0.75, 1])

# Create the solver
solver = casadi.Opti()

# Define the Bezier curve variables
control_points = [solver.variable(2, 1) for _ in range(NUM_CONTROL_POINTS)]

# Define the Bezier function
def bezier(i, n, t):
    binomial_coef = scipy.special.comb(n, i)
    return binomial_coef * (1 - t)**(n - i) * t**i

# Adds constraints
for j in range(NUM_INTERMEDIARY_POINTS):
    z = sum(control_points[i] * bezier(i, NUM_CONTROL_POINTS - 1, time_moments[j]) for i in range(NUM_CONTROL_POINTS))
    solver.subject_to(z == intermediary_points[j])

# Define the cost to be minimized
def cost(n):
    result = 0
    for i in range(n):
        for k in range(n):
            integral_result = scipy.integrate.quad(lambda t: bezier(i, n - 1, t) * bezier(k, n - 1, t), 0, 1)[0]
            result += casadi.mtimes(casadi.transpose(control_points[i + 1] - control_points[i]), control_points[k + 1] - control_points[k]) * (n ** 2) * integral_result
    return result

solver.minimize(cost(NUM_CONTROL_POINTS - 1))

# Define properties for the ipopt solver
solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
solver.solver('ipopt', solver_options)

# Run the solver
solution = solver.solve()

# Extract the optimized control points
optimized_control_points = [solution.value(control_points[i]) for i in range(NUM_CONTROL_POINTS)]


## Plotting

# Create the Bezier curve using the optimized control points
t_values = np.linspace(0, 1, 100)
curve_points = np.zeros((100, 2))
for j in range(100):
    z = sum(optimized_control_points[i] * bezier(i, NUM_CONTROL_POINTS - 1, t_values[j]) for i in range(NUM_CONTROL_POINTS))
    curve_points[j] = z

# Plot the intermediary points, the Bezier curve, and the optimized control points
fig, ax = plt.subplots()
ax.plot(intermediary_points[:, 0], intermediary_points[:, 1], 'ro', label='Intermediary points')
ax.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier curve')
ax.plot([p[0] for p in optimized_control_points], [p[1] for p in optimized_control_points], 'bo', label='Optimized control points')
ax.legend()
plt.title("Optimized control points")
plt.show()

### iii

# Define new functions to calculate z_d1 and z_d2
def calculate_first_derivative(control_points, n, t):
    z = sum([(control_points[i+1] - control_points[i]) * bezier(i, n-1, t) for i in range(n)])
    return n * z

def calculate_second_derivative(control_points, n, t):
    z = sum([(control_points[i+2] - 2 * control_points[i+1] + control_points[i]) * bezier(i, n-2, t) for i in range(n-1)])
    return n * (n - 1) * z

# Define new functions to calculate u_v and u_phi
def calculate_u_v(z1_derivative_1, z2_derivative_1):
    return (z1_derivative_1 ** 2 + z2_derivative_1 ** 2) ** 0.5

def calculate_u_phi(z1_derivative_1, z2_derivative_1, z1_derivative_2, z2_derivative_2):
    numerator = z2_derivative_2 * z1_derivative_1 - z2_derivative_1 * z1_derivative_2
    denominator = (z1_derivative_1 ** 2 + z2_derivative_1 ** 2) ** 1.5
    return casadi.atan(numerator / denominator)

# Define the range of t values to evaluate u_v and u_phi
t_values = np.linspace(0, 1, 1000)

# Evaluate u_v and u_phi for each t value
u_v_results = []
u_phi_results = []
for t in t_values:
    z_d1 = calculate_first_derivative(optimized_control_points, NUM_CONTROL_POINTS - 1, t)
    z_d2 = calculate_second_derivative(optimized_control_points, NUM_CONTROL_POINTS - 1, t)
    u_v_results.append(calculate_u_v(z_d1[0], z_d1[1]))
    u_phi_results.append(calculate_u_phi(z_d1[0], z_d1[1], z_d2[0], z_d2[1]))

# Plot u_v and u_phi
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

ax1.plot(t_values, u_v_results)
ax1.set_ylabel("u_v")
ax1.set_title("Velocity command")

ax2.plot(t_values, u_phi_results)
ax2.set_xlabel("Time")
ax2.set_ylabel("u_phi")
ax2.set_title("Steering angle command")

plt.show()