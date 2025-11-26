import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 50




# ============================================================
#       3D Molecular Dynamics with Velocity Verlet
# ============================================================

# Use reduced units here unless otherwise specified
sigma = 1 
eps = 1
m = 1

# initial positions and velocities
def initialization_3d(n,sigma):   # 3D , n = number of particles on each dimension
    """
    Initialize positions and velocities for a 3D MD simulation.

    Particles are placed on a cubic lattice and given random velocities
    with zero total momentum (center-of-mass frame).

    Parameters
    ----------
    n : int
        Number of particles in each dimension (total N = n^3).
    sigma : float
        Lennard-Jones sigma (used for lattice spacing).

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Initial positions on a cubic lattice.
    vel : ndarray of shape (N, 3)
        Initial velocities with zero net momentum.
    KE : ndarray of shape (N, 3)
        Kinetic energy contributions per component v^2 / 2 (with m = 1).
    N : int
        Total number of particles.
    """

    N = n**3    # total number of particles
    
    d =  10.229*sigma /864**(1/3) # [A]
    
    pos = np.zeros((N ,3))
    count = 0

    # Initialize positions
    for i1 in range( n ):
        for i2 in range( n ):
            for i3 in range( n ):
                pos[ count ] = d*i1 , d*i2 , d*i3
                count += 1
    
    vel = np.zeros((N,3))


    # Initialize velocities in random directions
    for atom in range(N):
        v_rand = np.random.rand()
        thita = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        

        # Spherical to Cartesian
        i = 0
        vel[atom , i] = v_rand*np.sin(thita)*np.cos(phi)        # velocity at x-axis
        vel[atom , i+1] = v_rand*np.sin(thita)*np.sin(phi)      # velocity at y-axis
        vel[atom , i+2] = v_rand*np.cos(thita)                  # velocity at z-axis


    
    # Remove center-of-mass velocity to work in the inertial frame
    for i in range(3):
        v_g = sum(vel[: , i]) / N        # V_g = sum ( m_i * v_i  ) / sum( m_i ) where m_i = 1 , inertial frame of reference

        vel[: , i] = vel[:,i] - v_g     # initial velocities for each atom at x,y,z axis

    


    KE = vel**2 / 2
        
    return pos , vel , KE, N 








def lj_force_3d(n, X , eps, sigma):
    """
    Compute LJ forces and potential energy for a given configuration X in 3D.

    Uses minimum image convention and the standard Lennard-Jones force.

    Parameters
    ----------
    n : int
        Number of particles per dimension (total N = n^3).
    X : ndarray of shape (N, 3)
        Particle positions.
    eps : float
        Lennard-Jones epsilon.
    sigma : float
        Lennard-Jones sigma.

    Returns
    -------
    forces : ndarray of shape (N, 3)
        Forces on each particle.
    V : float
        Total potential energy.
    """

    N = n**3
    
    d =  10.229*sigma /864**(1/3)
    L = N*d
    D = X[None ,:,:] - X[:, None ,:]
    D -= np.rint( D/L )*L

    D2 = np . sum( D**2 , axis =-1)     # equals to r^2 = x^2 + y^2 + z^2  
    DS2= np . divide ( sigma **2 , D2 , where=D2>0  )    # equals to sigma^2 / r^2 , put the condition D2>0 so that not to include calculations beteween the same particles    
    DS6 = DS2**3    # equals to {sigma^2 / r^2}^3

    V = 4*eps*np.sum( DS6 *( DS6 - 1) )/2

    forces = np.zeros((N,3))
    for element in range(N):
        for col in range(3):
            r = D[element, : , col]     # gives the distance at each axis between the elements
            r2 = D2[element]
            rs6 = DS6[element]
            F = 48*eps*rs6*r *(rs6-0.5)/ r2     # force at each element at each axis due to the rest of the elements
            
            for i in range(len(F)):
                    F[i] = np.nan_to_num(F[i])      # replace the nan values with zeros, I get nan values when I devide by 0 in the case that I compute the force between the same element
            F = sum(F)
            forces[element,col] = F   
    return forces , V




# Example: compute one force configuration for n=2
X , vel , KE, N = initialization_3d(2,sigma)

lj_force_3d(2 , X , eps ,sigma)





def verlet(n, dt, tmax):
    """
    Time integrate the 3D system using the Velocity Verlet algorithm.

    The LJ forces are computed at each step, and positions and velocities
    are updated accordingly.

    Parameters
    ----------
    n : int
        Number of particles per dimension (total N = n^3).
    dt : float
        Time step.
    tmax : float
        Maximum simulation time.

    Returns
    -------
    X : ndarray of shape (N, 3)
        Final positions after the last time step.
    vel : ndarray of shape (N, 3)
        Final velocities after the last time step.
    """

    X , vel, KE, N = initialization_3d(n, sigma)

    NumOfSteps = int(np.rint(tmax / dt))

    for i in range(1, NumOfSteps):
        
        # the following correspond to all N atoms
        # Current forces and potential
        F , V = lj_force_3d(n , X , eps, sigma)

        a = F 

        # Position update
        Xnew = X + vel * dt +  a * dt**2/2
        
        # New forces at updated positions
        Fnew , Vnew = lj_force_3d( n , Xnew , eps, sigma)
        a_new = Fnew

        # Velocity update
        vel_new = vel + ( a_new + a) *dt/2

        # Move to the new state
        X = Xnew
        vel = vel_new
           
    return X , vel










