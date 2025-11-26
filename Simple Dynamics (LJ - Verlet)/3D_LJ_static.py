import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 50


# ============================================================
#       Lennard-Jones potential and forces in 3D (static)
# ============================================================

## INITIAL CONDITIONS
def positions_3d(n):   # 3D , n = number of particles on each dimension
    
    """
    Generate a simple cubic lattice of particles in 3D.

    Parameters
    ----------
    n : int
        Number of particles along each spatial dimension.
        Total number of particles will be N = n^3.

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Array with particle positions in 3D.
        Particles are placed on a cubic lattice with spacing d.
    N : int
        Total number of particles N = n^3.
    """
    
    N = n**3    # total number of particles
    sigma = 3.405
    d =  10.229*sigma /864**(1/3) # Lattice spacing (in Angstroms); 10.229*sigma divided by cubic root of 864

    
    pos = np.zeros((N ,3))
    count = 0

    # Fill positions on a regular cubic grid
    for i1 in range( n ):
        for i2 in range( n ):
            for i3 in range( n ):
                pos[ count ] = d*i1 , d*i2 , d*i3
                count += 1
    return pos , N




def lj_3d(n):

    """
    Compute Lennard-Jones potential energy and forces for a 3D system
    of N = n^3 particles placed on a cubic lattice.

    Uses minimum image convention (periodic boundary conditions).

    Parameters
    ----------
    n : int
        Number of particles along each spatial dimension.

    Returns
    -------
    V : float
        Total Lennard-Jones potential energy of the system.
    forces : ndarray of shape (N, 3)
        Force on each particle (Fx, Fy, Fz) due to all others.
    """

    X , N = positions_3d(n)
    eps = 1.654*10**(-21)    # Lennard-Jones epsilon in Joules
    sigma = 3.405     # Lennard-Jones sigma in Angstroms
    d =  10.229*sigma /864**(1/3)
    L = N*d # effective box length (1D) used for the minimum image convention
    
    # Pairwise displacement tensor: D[i, j, :] = X[i] - X[j]
    D = X[None ,:,:] - X[:, None ,:]

    # Apply minimum image convention: wrap displacements into the box [-L/2, L/2]
    D -= np.rint( D/L )*L

    # Squared distances r^2 for all pairs shape (N, N)
    D2 = np . sum( D**2 , axis =-1)     # equals to r^2 = x^2 + y^2 + z^2  

    # sigma^2 / r^2, ignoring self-interactions (where D2 = 0)
    DS2= np . divide ( sigma **2 , D2 , where=D2>0  )  

    # (sigma^2 / r^2)^3 = (sigma / r)^6  
    DS6 = DS2**3    


    # Total Lennard-Jones potential energy:
    # V = sum over all pairs of 4*eps[(sigma/r)^12 - (sigma/r)^6]
    # Divide by 2 to avoid double-counting i-j and j-i
    V = 4*eps*np.sum( DS6 *( DS6 - 1) )/2  
    

    # Forces on each particle
    forces = np.zeros((N,3))
    for element in range(N):
        for col in range(3):

            # r is the displacement along a single axis to all other particles
            r = D[element, : , col]     
            r2 = D2[element]
            rs6 = DS6[element]

            # F_ij = 48*eps * (sigma/r)^6 * [(sigma/r)^6 - 0.5] * (r_ij / r^2)
            # This is the component of the force along the given axis
            # Force at each element at each axis due to the rest of the elements
            F = 48*eps*rs6*r *(rs6-0.5)/ r2    

            for i in range(len(F)):
                    F[i] = np.nan_to_num(F[i])      # replace the nan values with zeros, I get nan values when I devide by 0 in the case that I compute the force between the same element
            F = sum(F)
            forces[element,col] = F   

    return V, forces