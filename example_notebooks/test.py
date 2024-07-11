import numpy as np
import matplotlib.pyplot as plt

def simulate_convection(CFL, Tfinal):
    # Domain length
    a = 0
    b = 6
    # Number of nodes
    N = 100
    # Mesh step size
    dx = (b - a) / (N - 1)
    # Transport velocity
    c = 2
    x = np.linspace(a, b, N)

    # Initial condition function
    def g(x):
        x = x % b
        if 1/2 <= x <= 3/2:
            return 1
        else:
            return 0

    # Initialize the initial condition
    u = np.zeros(N)
    for i in range(N):
        u[i] = g(x[i])

    # Compute time step to ensure stability
    dt = CFL * dx / abs(c)
    lamda = c * dt / dx

    unewp = np.zeros(N)
    unewx = np.zeros(N)

    temps = 0

    while temps < Tfinal:
        for i in range(N):
            unewx[i] = g(x[i] - c * temps)
        for i in range(1, N-1):
            flux_left = max(0, c) * u[i-1] + min(c, 0) * u[i]
            flux_right = max(0, c) * u[i] + min(c, 0) * u[i+1]
            unewp[i] = u[i] - lamda * (flux_right - flux_left)  # Upwind scheme
        # Neumann boundary conditions (Zero derivatives)
        unewp[0] = u[N-1]
        unewp[N-1] = unewp[N-2]
        temps += dt
        u = unewp.copy()
        ux = unewx.copy()

    return x, u, ux

# Define values for CFL and Tfinal
CFLs = [0.4, 0.8, 1, 1.1]
Tfinals = [0.5, 1.5]

# Create a figure and subplots
fig, axs = plt.subplots(len(Tfinals), len(CFLs), figsize=(12, 10), sharex=True, sharey=True)

# Loop over each combination of Tfinal and CFL
for ti, Tfinal in enumerate(Tfinals):
    for tj, CFL in enumerate(CFLs):
        x, u, ux = simulate_convection(CFL, Tfinal)
        axs[ti, tj].plot(x, ux, '-r', label='Exact Solution')
        axs[ti, tj].plot(x, u, '-b', label='Numerical Solution')
        axs[ti, tj].set_title(f"Tfinal = {Tfinal}, CFL = {CFL}")
        axs[ti, tj].set_ylim(-1.5, 1.5)  # Limit the y-axis

        axs[ti, tj].grid()

# Add legend to the first subplot only to avoid repetition
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

fig.suptitle("Simulation d'un problème de convection 1D par Volumes Finis")
fig.text(0.5, 0.04, 'x', ha='center')
fig.text(0.04, 0.5, 'u(x, t)', va='center', rotation='vertical')
plt.tight_layout()
plt.show()
