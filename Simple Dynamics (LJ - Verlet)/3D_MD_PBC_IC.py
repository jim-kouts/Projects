import numpy as np
import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import plotly.graph_objects as go
from numba import njit


# ------------------------------------------------------------
# 1. Physical constants and simulation parameters
# ------------------------------------------------------------

n=5                             # number of particles along each dimension (total N = n^3)
m= 39.948                       # [amu] for argon
m *= 931.5                      # [MeV / c^2]
m *= 1e6                        # [eV / c^2]
m /=  (299792458**2)            # [eV * s^2/m^2], E = mc^2
m /=  1e20                      # [eV * s^2/A^2] since 1 m = 1e10 Å


# Lennard-Jones parameters for Argon
eps=1.654e-21 * 6.24150907e18   # [eV]           
sigma = 3.405                   # [A]

kb = 8.617333262e-5             # [ eV/K ]                             

# Initial lattice spacing
d =  10.229*sigma /864**(1/3)   # [ A ]

# Box length (cubic box with periodic boundary conditions)
L= n*d

# Target starting temperature
Tstart = 95                     #[K]



# ------------------------------------------------------------
# 2. Initialization of positions and velocities (3D)
# ------------------------------------------------------------


@njit()
def initialization_3d(n,d):   
    """
    Initialize a simple cubic lattice with n x n x n particles
    and assign random velocities with random orientations.

    Parameters
    ----------
    n : int
        Number of lattice sites per dimension.
    d : float
        Lattice spacing [Å].

    Returns
    -------
    pos : (N, 3) array
        Particle positions in 3D.
    vel : (N, 3) array
        Particle velocities in 3D (arbitrary magnitudes).
    N : int
        Total number of particles (n^3).
    """

    N = n**3  # total number of particles    

    # With numba, use np.empty instead of np.zeros
    pos = np.empty((N, 3)) 
    vel = np.empty((N, 3))
    count = 0

    # Fill positions on a cubic grid: (i1*d, i2*d, i3*d)
    for i1 in range( n ):
        for i2 in range( n ):
            for i3 in range( n ):
                pos[ count ] = d*i1 , d*i2 , d*i3
                count += 1
    

    # Initialize velocities with random magnitudes and directions
    for atom in range(N):
        v_rand = np.random.rand()           # random speed
        thita = np.random.uniform(0, np.pi) # polar angle
        phi = np.random.uniform(0, 2*np.pi) # azimuthal angle
        

        # Convert spherical to Cartesian components
        i = 0
        vel[atom , i] = v_rand*np.sin(thita)*np.cos(phi)        # velocity at x-axis
        vel[atom , i+1] = v_rand*np.sin(thita)*np.sin(phi)      # velocity at y-axis
        vel[atom , i+2] = v_rand*np.cos(thita)                  # velocity at z-axis

    
    
        
    return pos , vel , N 






# ------------------------------------------------------------
# 3. Velocity utilities: remove drift, scale temperature, KE
# ------------------------------------------------------------



@njit
def drift_vel(vel):
    """
    Remove center-of-mass velocity so that total momentum is zero.

    vel : (N,3) array
        Velocities of all particles.

    Uses global N (set after initialization_3d is called).
    """
    v_gx = 0.0
    v_gy = 0.0
    v_gz = 0.0

    # Compute average velocity in each direction
    for i in range(N):
        v_gx += vel[i, 0]
        v_gy += vel[i, 1]
        v_gz += vel[i, 2]

    v_gx /= N
    v_gy /= N
    v_gz /= N


    # Subtract COM velocity from each particle
    for i in range(N):
        vel[i, 0] -= v_gx
        vel[i, 1] -= v_gy
        vel[i, 2] -= v_gz
        
    return vel


@njit
def scale_vel(vel, Tstart):
    """
    Rescale velocities so that the instantaneous temperature
    equals Tstart, using equipartition:

        T = m * sum_i |v_i|^2 / (3 * kb * N)

    Returns
    -------
    vel_scaled : (N,3) array
        Rescaled velocities.
    T : float
        Temperature before rescaling.
    """
    
    v2_sum = 0.0
    for i in range(N):
        vx = vel[i, 0]
        vy = vel[i, 1]
        vz = vel[i, 2]
        v2_sum += vx*vx + vy*vy + vz*vz
    
    
    # Instantaneous temperature before scaling
    T  = (m * v2_sum) / (3.0 * kb * N)

    # Scaling factor
    vel_scaled = vel * np.sqrt(Tstart / T )
    
    return vel_scaled , T





@njit
def kinetic_en(vel):
    
    """
    Compute total kinetic energy:

        KE = 1/2 m sum_i |v_i|^2

    Returns
    -------
    KE : float
        Total kinetic energy [eV].
    """

    v2_sum = 0.0
    for i in range(N):
        vx = vel[i, 0]
        vy = vel[i, 1]
        vz = vel[i, 2]
        v2_sum += vx*vx + vy*vy + vz*vz

    KE = 0.5 * m * v2_sum

    return KE    #     [eV]






# ------------------------------------------------------------
# 4. Build initial configuration and visualize the lattice
# ------------------------------------------------------------



pos , vel , N = initialization_3d(n,d)

# 3D scatter of initial positions
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='markers', marker=dict(size=3)))
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X-axis [\u212B]'  ),       
        yaxis=dict(title='Y-axis [\u212B]'),
        zaxis=dict(title='Z-axis [\u212B]'),
    )
)
fig.show()





# ------------------------------------------------------------
# 5. Lennard-Jones potential and forces (numba versions)
# ------------------------------------------------------------




@njit
def lj_pot(X, eps, sigma, L):
    """
    Total Lennard-Jones potential energy with PBC (minimum image),
    summed over i<j:

        V = sum_{i<j} 4ε [ (σ / r_ij)^12 - (σ / r_ij)^6 ]

    X : (N,3) positions
    """

    N = X.shape[0]
    V = 0.0
    sig2 = sigma * sigma

    for i in range(N - 1):
        xi0 = X[i, 0]
        xi1 = X[i, 1]
        xi2 = X[i, 2]
        for j in range(i + 1, N):

            # Displacement between i and j
            dx = xi0 - X[j, 0]
            dy = xi1 - X[j, 1]
            dz = xi2 - X[j, 2]

            # Minimum-image convention (wrap into [-L/2, L/2])
            dx -= L * np.rint(dx / L)
            dy -= L * np.rint(dy / L)
            dz -= L * np.rint(dz / L)


            # Squared distance
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > 0.0:
                inv_r2 = sig2 / r2                 # (σ/r)^2
                inv_r6 = inv_r2 * inv_r2 * inv_r2  # (σ/r)^6

                V += 4.0 * eps * (inv_r6 * (inv_r6 - 1.0))

    return V



@njit
def lj_force(X, eps, sigma, L):

    """
    Compute Lennard-Jones forces on all particles with PBC
    using minimum image.

    Returns
    -------
    forces : (N,3) array
        Force on each particle [eV/Å].
    """

    N = X.shape[0]
    forces = np.zeros((N, 3))
    sig2 = sigma * sigma

    for i in range(N - 1):
        xi0 = X[i, 0]
        xi1 = X[i, 1]
        xi2 = X[i, 2]
        for j in range(i + 1, N):
            dx = xi0 - X[j, 0]
            dy = xi1 - X[j, 1]
            dz = xi2 - X[j, 2]

            # Minimum image
            dx -= L * np.rint(dx / L)
            dy -= L * np.rint(dy / L)
            dz -= L * np.rint(dz / L)

            r2 = dx*dx + dy*dy + dz*dz
            if r2 > 0.0:
                inv_r2 = sig2 / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2

                # magnitude factor: 48*eps*inv_r6*(inv_r6-0.5)/r2
                fac = 48.0 * eps * inv_r6 * (inv_r6 - 0.5) / r2

                fx = fac * dx
                fy = fac * dy
                fz = fac * dz

                # Add to i, subtract from j (Newton's third law)
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[i, 2] += fz

                forces[j, 0] -= fx
                forces[j, 1] -= fy
                forces[j, 2] -= fz

    return forces






# ------------------------------------------------------------
# 6. Velocity-Verlet integrator (3D)
# ------------------------------------------------------------



@njit
def verlet_3d(X, vel, dt, L):
    """
    One Velocity-Verlet step for all particles:

        X_new = X + v*dt + (1/2) a*dt^2
        v_new = v + (1/2)(a + a_new)*dt

    where a = F/m. Periodic boundaries are applied to positions.
    """


    # Forces and accelerations at time t
    F = lj_force(X, eps,sigma, L)
    a = F / m  
    
    # Update positions
    Xnew = X + vel * dt +  a * dt**2/2
    Xnew -= np.rint( Xnew/L )*L

    # Forces and accelerations at new positions
    Fnew = lj_force( Xnew,eps,sigma, L)
    a_new = Fnew/m

    # Update velocities
    vel_new = vel + ( a_new + a) *dt/2

          
    return Xnew , vel_new





# ------------------------------------------------------------
# 7. Time integration settings
# ------------------------------------------------------------


dt=  1e-16    # time step [s]
tmax = 5e-12  # total simulation time [s]

NumOfSteps = int(np.rint(tmax / dt))


# Equilibration: first fraction of the run where velocities are rescaled
equil_frac = 0.2                    # first 20% of the run = equilibration
equil_steps = max(1, int(equil_frac * NumOfSteps))


# Number of times you want to rescale during equilibration
n_rescales = 50                     # how many times you want to rescale in equil.
rescale_every = max(1, equil_steps // n_rescales)



# ------------------------------------------------------------
# 8. Allocate arrays to store time evolution
# ------------------------------------------------------------


x_t   = np.zeros((N, 3, NumOfSteps))
vel_t = np.zeros((N, 3, NumOfSteps))
V_t   = np.zeros(NumOfSteps)
KE_t  = np.zeros(NumOfSteps)
H_t   = np.zeros(NumOfSteps)
Temp  = np.zeros(NumOfSteps)


# ------------------------------------------------------------
# 9. Initial velocity processing: remove drift, scale to Tstart
# ------------------------------------------------------------

# Remove center-of-mass drift
vel = drift_vel(vel)

# Scale velocities to match target temperature
vel_scaled , Tnew  = scale_vel(vel , Tstart)


# Initial values t = 0
x_t[:, :, 0] = pos
V_t[0] = lj_pot(pos ,eps,sigma, L)
vel_t[:, :, 0] = vel_scaled
KE_t[0] = kinetic_en(vel_scaled)     
H_t[0] = KE_t[0]  +  V_t[0]    # H = Sum( KE ) + V   
Temp[0] = Tnew

# These are the state variables fed into the integrator
x_prev = pos
vel_prev = vel


# ------------------------------------------------------------
# 10. Main MD loop with early rescaling (equilibration)
# ------------------------------------------------------------

for i in range( 1, NumOfSteps):

    # Step forward one dt with Velocity-Verlet
    Xnew , vel_new = verlet_3d(x_prev, vel_prev ,dt, L)

    # Rescale velocities during equilibration phase at fixed intervals
    if i <= equil_steps and (i % rescale_every == 0):

        vel_new, Tnew = scale_vel(vel_new, Tstart)     
        Temp[i] = Tnew



    # Store positions, energies, velocities at current time step
    x_t[:, :, i] = Xnew   
    V_t[i] = lj_pot(Xnew ,eps,sigma, L)
    vel_t[:, :, i] = vel_new   
    KE_t[i] = kinetic_en(vel_new)      
    H_t[i]= KE_t[i]  + V_t[i]     # H = Sum( KE ) + V 

    # Update previous state for next step
    x_prev = Xnew
    vel_prev = vel_new
    

t_val = np.linspace(0 , tmax, NumOfSteps)



# ------------------------------------------------------------
# 11. Diagnostic plots: T(t), V(t), KE(t), H(t)
# ------------------------------------------------------------


# Temperature vs time
plt.figure()
plt.plot(t_val, Temp, label = f'$T(t)$ , dt={dt}' , linewidth=1.0 )
plt.xlabel("time [s]")
plt.ylabel(" Temperature [ K ]")
plt.legend()
# plt.savefig(f'Temp_{dt}.png')


# Potential energy vs time
plt.figure()
plt.plot(t_val, V_t , label = f'$V(t)$ , dt={dt}' )
plt.xlabel("time [s]")
plt.ylabel("Potential [eV]")
plt.legend()
# plt.savefig(f'V_t{dt}.png')



# Kinetic energy vs time
plt.figure()
plt.plot(t_val, KE_t,label= f'$KE(t)$ , dt={dt}'  )
# plt.plot(t_val, KE_t2 ,label= f'$KE_2(t)$ , dt={dt}')
plt.xlabel("time [s]")
plt.ylabel("Kinetic energy [eV] ")
plt.legend()
# plt.savefig(f'KE_t{dt}.png')


# H, KE, and V on the same plot
plt.figure()
plt.plot(t_val, H_t ,label= f'$H(t) $, dt={dt}' , linewidth=1.0 )
plt.plot(t_val, KE_t ,label= f'$KE(t)$ , dt={dt}', linewidth=1.0)
plt.plot(t_val, V_t ,label= f'$V(t)$ , dt={dt}', linewidth=1.0)
plt.xlabel("time [s]")
plt.ylabel(" H , KE , V [eV]")
plt.legend()
# plt.savefig(f'H_V_KE_{dt}.png')



# Total energy alone vs time
plt.figure()
plt.plot(t_val, H_t , label= f'$H(t) $, dt={dt}')
plt.xlabel("time [s]")
plt.ylabel(" H [eV]")
plt.legend()
# plt.savefig(f'H_t{dt}.png')
