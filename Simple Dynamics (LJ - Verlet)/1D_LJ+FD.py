import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 50



# ============================================================
#   Lennard-Jones potential in 1D + finite-difference check
# ============================================================

## INITIAL CONDITIONS
def positions_1d(n):   # n = number of particles on each dimension
    """
    Generate 1D lattice positions for n particles, embedded in 3D array.

    Parameters
    ----------
    n : int
        Number of particles in 1D.

    Returns
    -------
    pos : ndarray of shape (N, 3)
        Positions for particles, varying only in x-direction.
    N : int
        Total number of particles (N = n).
    """
    
    N = n**1    # total number of particles
    sigma = 3.405   #  [A]
    d =  10.229*sigma /864**(1/3) #  [A]
    
    pos = np.zeros((N ,3))
    count = 0
    
    # Particles along x-axis only, y = z = 0
    for i1 in range( n ):
        pos[ count ] = d*i1 
        count += 1
    return pos , N



def lj_1d(n, X):    # X is the distance between each atom
    """
    1D Lennard-Jones potential and force between 2 atoms separated by distance X.

    Parameters
    ----------
    n : int
        Number of particles (used here as 2).
    X : float or ndarray
        Separation distance(s) between the two atoms.

    Returns
    -------
    V : float or ndarray
        Lennard-Jones potential energy (per pair; divided by 2).
    F : float or ndarray
        Analytic expression for the force magnitude between the two atoms.
    """

    N = n**1
    eps = 1.654*10**(-21)    # in [J]
    sigma = 3.405     # in [A]
    

    r = X
    r2 = r**2
    rs2= np . divide ( sigma **2 , r2  )    # equals to sigma^2 / r^2     
    rs6 = rs2**3    # equals to {sigma^2 / r^2}^3

    V = 4*eps*( rs6 *( rs6 - 1) )/2     # [J]   devided by 2 to cancel out the contributions from the same pair of atoms
    
    # since we have 2 atoms, Fij = -Fji so I compouted only one F so Fij = 48*eps* (sigma/r)^6 * [(sigma/r)**6 - 0.5] / r
    
    F = 48*eps*rs6 *(rs6-0.5)/ r    # [J/A = kg*A/s^2]


    return V, F




# ---- Compute and compare analytical vs numerical derivative of V ----

x_val = np.linspace(3,10,50)

V, F = lj_1d(2, x_val)

# Finite difference approximation for dV/dx (simple forward difference)

Dv =[]
for i in range(len(x_val) -1 ):     

    dv = V[i+1] - V[i]
    
    Dv.append(dv)


newf = np.gradient(V, x_val[1] - x_val[0] )     





# Stack arrays horizontally for saving
data = np.column_stack((x_val, V, F, newf))

# Save as a CSV file
np.savetxt("data.csv", data, delimiter=" , ", header="x_val,V,F,newf", comments='')

# Load the data from the CSV file
loaded_data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

# Split the data into separate variables
x_val = loaded_data[:, 0]
V = loaded_data[:, 1]
F = loaded_data[:, 2]
newf = loaded_data[:, 3]



# ---- Plot: analytical force vs separation ----
plt.figure()
plt.plot(x_val, -F)
plt.grid(True)
plt.xlabel("separation on x-axis")
plt.ylabel("Force [kg*\u212B/s^2]")
# plt.savefig(f'F-1d.png')


# ---- Plot: potential vs separation ----
plt.figure()
plt.plot(x_val, V)
plt.grid(True)
plt.xlabel("separation x-axis")
plt.ylabel("LJ-potential [J]")
# plt.savefig(f'V-1d.png')


# ---- Combined plot: force, potential, and numerical derivative ----
plt.figure()
plt.plot(x_val, -F, label="Force", color="blue" , linewidth = 1)
plt.plot(x_val, V, label="LJ-Potential", color="red",linewidth = 1)
plt.plot(x_val, newf,label="dV/dx", color="black",linewidth = 1)
plt.grid(True)
plt.xlabel("Separation on x-axis")
plt.ylabel("F(x) / V(x) / (dV/dx)")
plt.legend()
# plt.savefig(f'F-V-dV-1d.png')


# ---- Plot: numeric derivative alone ----
plt.figure()
plt.plot(x_val, newf)
plt.grid(True)
plt.xlabel("separation on x-axis")
plt.ylabel("Force [kg *\u212B/s^2]")
# plt.savefig(f'dVprime-1d.png')


plt.show()