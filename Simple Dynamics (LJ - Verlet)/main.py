import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 50


# ============================================================
# TASK 1: Lennard-Jones potential and forces in 3D (static)
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







# ============================================================
# TASK 2: Lennard-Jones potential in 1D + finite-difference check
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






# ============================================================
# TASK 3: 3D Molecular Dynamics with Velocity Verlet
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





# ============================================================
# TASK 4: 1D two-particle MD with energy monitoring
# ============================================================

def initialization_1d(n):   # 1D , n = number of particles on each dimension
    """
    Initialize a simple 1D system of two particles.

    Particle 1 is at x = 0, particle 2 is at x = 4 Å,
    both with zero initial velocity.

    Parameters
    ----------
    n : int
        Number of particles (here used as 2).

    Returns
    -------
    pos1, pos2 : float
        Initial positions of the two particles.
    vel1, vel2 : float
        Initial velocities of the two particles.
    N : int
        Total number of particles (N = n).
    """
    
    N = n**1    # total number of particles

    pos1 = 0
    pos2 = pos1 + 4  # [A]
     
    vel1 = 0    # [A/ s]
    vel2 = 0
  
    return pos1 ,pos2 , vel1, vel2 , N 




pos1 ,pos2 , vel1, vel2 , N = initialization_1d(2)


m =1 #39.948 # [amu] for argon
eps = 1     #1.654e-21 * 6.24150907e18   [eV]
sigma =1    #3.405  [A]




def kin_en(vel1 , vel2, m):
    """
    Compute kinetic energy of two particles in 1D.

    Parameters
    ----------
    vel1, vel2 : float
        Velocities of the two particles.
    m : float
        Mass of each particle.

    Returns
    -------
    KE1, KE2 : float
        Kinetic energies of particle 1 and 2.
    """

    KE1 = m*vel1**2 / 2     
    KE2 = m*vel2**2 / 2

    return KE1 , KE2    # [ (A/s)^2 ]  or [ eV ] if I add m


def potential(pos1 , pos2 , eps, sigma):
    """
    Lennard-Jones potential energy between two particles in 1D.

    Parameters
    ----------
    pos1, pos2 : float
        Positions of the two particles.
    eps, sigma : float
        LJ parameters.

    Returns
    -------
    V : float
        Lennard-Jones potential energy.
    """
    
    D = pos1 - pos2
    D2 = D**2     # equals to r^2 = x^2  
    DS2=  sigma **2  / D2           
    DS6 = DS2**3    # equals to {sigma^2 / r^2}^3

    V = 4*eps*DS6* (DS6 - 1)
   
    return V        #   [ 1/ A^6 ] since only D has units at this point




def Force(pos1 , pos2, eps, sigma):
    """
    Pairwise forces on two particles in 1D from the LJ potential.

    Parameters
    ----------
    pos1, pos2 : float
        Positions of the two particles.
    eps, sigma : float
        LJ parameters.

    Returns
    -------
    F1, F2 : float
        Forces acting on particle 1 and 2, respectively.
    """
    
    D = pos1 - pos2
    D2 = D**2     # equals to r^2 = x^2  
    DS2= sigma **2 / D2          
    DS6 = DS2**3    # equals to {sigma^2 / r^2}^3

    F = 48*eps*DS6 *(DS6-0.5)/ D     
    
    F1 = F
    F2 = -F

    return F1, F2   # [ 1 / A^7]




def verlet_1d(n,dt,tmax , eps, sigma):
    """
    Velocity Verlet integration for two particles in 1D with LJ interaction.

    The function writes the trajectory and energies to a CSV file named
    'data_<dt>.csv'.

    Parameters
    ----------
    n : int
        Number of particles (here 2).
    dt : float
        Time step.
    tmax : float
        Total simulation time.
    eps, sigma : float
        LJ parameters.

    Returns
    -------
    None
        Data are saved directly to CSV.
    """
    pos1 ,pos2 , vel1, vel2 , N =initialization_1d(2)
    
    NumOfSteps = int(np.rint(tmax / dt))



    filename = f"data_{dt}.csv"
    with open(filename, 'w') as f:

        # Write the header
        f.write("t_val, x_t1, x_t2, vel_t1, vel_t2, KE_t1, KE_t2, H_t, V_t\n")


        
        # Initial values 
        V_t = potential(pos1 , pos2,eps , sigma) 


        KE_t1 = kin_en(vel1 , vel2, m) [0]     # append only the KE1 value that the kin_en function returns
        KE_t2 = kin_en(vel1 , vel2 , m) [1]      # append only the KE2 value that the kin_en function returns

        H_t = KE_t1  + KE_t2  + V_t     # H = Sum( KE ) + V
        t = 0.0

        f.write(f"{t},{pos1},{pos2},{vel1},{vel2},{KE_t1},{KE_t2},{H_t},{V_t}\n")


        # ---- Velocity Verlet integration loop ----

        for i in range( 1, NumOfSteps):
            
            F1, F2 = Force(pos1 , pos2,eps , sigma ) 
            
            a1 = F1/m      
            a2 = F2 /m
            pos_new1 = pos1 + vel1 * dt + a1 * dt**2 / 2    
            pos_new2 = pos2 + vel2 * dt + a2 * dt**2 / 2

            # Forces at new positions
            a_new1 , a_new2 = Force(pos_new1 , pos_new2,eps , sigma)
            

            vel_new1  =  vel1 + (a_new1 + a1)*dt / 2
            vel_new2  =  vel2 + (a_new2 + a2)*dt / 2
            

            V_t= potential(pos_new1 , pos_new2,eps , sigma) 

            

            KE_t1=kin_en(vel_new1 , vel_new2, m) [0]     
            KE_t2= kin_en(vel_new1 , vel_new2, m) [1] 

            H_t=KE_t1 + KE_t2  + V_t    # H = Sum( KE ) + V , KE_t1[-1]+ KE_t2[-1]
            
            t = i * dt


            f.write(f"{t},{pos_new1},{pos_new2},{vel_new1},{vel_new2},{KE_t1},{KE_t2},{H_t},{V_t}\n")

            pos1 = pos_new1    
            pos2 = pos_new2
            vel1 = vel_new1
            vel2 = vel_new2
              



# ---- Run 1D Verlet for different time steps and plot results ----

dt= [ 0.1 , 0.01 , 0.001 ]
tmax = 100

for h in dt:
    
    verlet_1d(2 , h , tmax ,eps , sigma)


    # Load the data from the CSV file
    loaded_data = np.loadtxt(f"data_{h}.csv", delimiter=",", skiprows=1)

    # Split the data into separate variables
    t_val = loaded_data[:, 0]
    x_t1 = loaded_data[:, 1]
    x_t2 = loaded_data[:, 2]
    vel_t1 = loaded_data[:, 3]
    vel_t2 = loaded_data[:, 4]
    KE_t1 = loaded_data[:, 5]
    KE_t2 = loaded_data[:, 6]
    H_t = loaded_data[:, 7]
    V_t = loaded_data[:, 8]




    # Trajectory plots
    plt.figure()
    plt.plot(t_val, x_t1 , label= f'$x_1(t)$ , dt={h}')
    plt.plot(t_val, x_t2 , label= f'$x_2(t)$, dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel(" Position [ \u212B ]")
    plt.legend()
    # plt.savefig(f'x_t12_{h}.png')

    

    plt.figure()
    plt.plot(t_val, vel_t1 ,label= f'$u_1(t)$, dt={h}' )
    plt.plot(t_val, vel_t2 ,label= f'$u_2(t)$, dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [ \u212B / sec]")
    plt.legend()
    # plt.savefig(f'u_t12_{h}.png')

    
    # Energies
    plt.figure()
    plt.plot(t_val, V_t , label = f'$V(t)$ , dt={h}' , linewidth=1.0 )
    plt.plot(t_val, KE_t1 ,label= f'$KE_1(t)$, dt={h}', linewidth=1.0 )
    plt.xlabel("time [s]")
    plt.ylabel(" V , KE [eV]")
    plt.legend()
    # plt.savefig(f'V_KE_{h}.png')

    
    plt.figure()
    plt.plot(t_val, V_t , label = f'$V(t)$ , dt={h}' )
    plt.xlabel("time [s]")
    plt.ylabel("Potential [eV]")
    plt.legend()
    # plt.savefig(f'V_{h}.png')

    
    plt.figure()
    plt.plot(t_val, KE_t1 ,label= f'$KE_1(t)$ , dt={h}'  )
    plt.plot(t_val, KE_t2 ,label= f'$KE_2(t)$ , dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel("Kinetic energy [eV] ")
    plt.legend()
    # plt.savefig(f'KE_12_{h}.png')

    

    plt.figure()
    plt.plot(t_val, H_t ,label= f'$H(t) $, dt={h}' , linewidth=1.0 )
    plt.plot(t_val, KE_t1 ,label= f'$KE_1(t)$ , dt={h}', linewidth=1.0)
    plt.plot(t_val, V_t ,label= f'$V(t)$ , dt={h}', linewidth=1.0)
    plt.xlabel("time [s]")
    plt.ylabel(" H , KE , V [eV]")
    plt.legend()
    # plt.savefig(f'H_V_KE_{h}.png')

    
    plt.figure()
    plt.plot(t_val, H_t , label= f'$H(t) $, dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel(" H [eV]")
    plt.legend()
    # plt.savefig(f'H_{h}.png')

    

    plt.show()




