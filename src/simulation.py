import numpy as np
import matplotlib.pyplot as plt

def simulate_heat_conduction(alpha, out_channels=2, t_diff=5, dt=0.01, L=0.1, T_top=20, T_bottom=400, decimals=5):
    n = alpha.shape[0]
    dx = L / (n - 1) # grid spacing
    
    # (1/dx**2 + 1/dy**2 + 1/dz**2) * dt * alpha_Cu (CFL/stability condition)
    assert 3*dt*alpha.min()/dx**2 < 0.5, 'Courant-Friedrichs-Lewy condition is not met'
    
    T_cube = np.ones((n, n, n)) * T_top # initial temperature array
    
    t_final = out_channels * t_diff
    n_iter = int(t_final / dt) # number of iterations
    
    output = np.zeros((out_channels,n,n))
    
    # Loop over time
    for i in range(n_iter):
        # Apply boundary conditions
        T = np.pad(T_cube, 1, 'edge') # insulated sides
        T[0,:,:] = T_bottom
        T[-1,:,:] = T_top

        # Calculate the second-order spatial derivatives
        # (d^2 T)/(dx^2) ≈ [T(i+1,j,k) - 2T(i,j,k) + T(i-1,j,k)] / (Δx)^2
        d2T_dx2 = (T[2:,1:-1,1:-1] - 2*T_cube + T[:-2,1:-1,1:-1]) / dx**2
        d2T_dy2 = (T[1:-1,2:,1:-1] - 2*T_cube + T[1:-1,:-2,1:-1]) / dx**2
        d2T_dz2 = (T[1:-1,1:-1,2:] - 2*T_cube + T[1:-1,1:-1,:-2]) / dx**2

        # Calculate the temperature at the next time step
        T_cube += alpha * dt * (d2T_dx2 + d2T_dy2 + d2T_dz2)
        
        t_elapsed = (i+1)*dt
        if t_elapsed % t_diff == 0:
            output[int(t_elapsed/t_diff)-1] = T_cube[-1]
            
    return output.round(decimals)

def plot_top_temperature(T, channel=-1, figsize=(7,7), ax=None):
    if ax == None:
        _, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(T[channel], cmap='coolwarm')
    ax.set_title('Temperature distribution on the top surface')
    plt.show()