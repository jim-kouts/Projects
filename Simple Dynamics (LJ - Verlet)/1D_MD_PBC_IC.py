import numpy as np
import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import plotly.graph_objects as go

# ----------------------------
# 1. Physical parameters
# ----------------------------

n=2

# Argon mass in simulation units

m=39.948                #  [amu]             
m *= 931.5              #  [MeV]     [MeV / c^2]
m *= 1e6                #  [eV]      [eV / c^2]
m /=  (299792458**2)    #  [eV * s^2/m^2]    E = mc^2
m /=  1e20              #  [eV * s^2/A^2]


# Lennard-Jones parameters for Argon
eps= 1.654e-21* 6.24150907e18  # [eV]           
sigma = 3.405                  # [A]

# Lattice spacing / characteristic distance between particles
d =  10.229*sigma /864**(1/3)  # [A]

# Box length (for periodic boundary conditions)
L= n*d



# ----------------------------
# 2. Initialization routines
# ----------------------------

def initialization_1d(n):   # 1D , n = number of particles on each dimension
    """
    1D system: initialize two particles in a 1D box of length L.

    Parameters
    ----------
    n : int
        "Number of particles per dimension".

    Returns
    -------
    pos1, pos2 : float
        Initial positions of the two particles (in Å).
    vel1, vel2 : float
        Initial velocities of the two particles (in Å/s).
    N : int
        Total number of particles (here always 2).
    """
    
    # Total number of particles
    N = n**1    

    # Place the two particles at fixed fractions of the box length
    pos1 = 0.1*L
    pos2 = pos1 + 0.4*L  # [A]
     
    # Start both particles at rest
    vel1 = 0             # [A/ s]
    vel2 = 0

    
    return pos1 ,pos2 , vel1, vel2 , N 



# ----------------------------
# 3. Energies and potential
# ----------------------------

def kin_en(vel1 , vel2, m):
    """
    Compute kinetic energy of the two particles.

    KE_i = 1/2 m v_i^2

    Returns
    -------
    KE1, KE2 : floats
        Kinetic energy of each particle in eV.
    """

    KE1 = m*vel1**2 / 2     
    KE2 = m*vel2**2 / 2

    return KE1 , KE2    #  [ eV ]



def potential(pos1 , pos2 , eps, sigma,L ):
    """
    Lennard-Jones potential between the two particles in 1D
    with periodic boundary conditions (minimum image).

    V(r) = 4ε [ (σ/r)^12 − (σ/r)^6 ]

    Parameters
    ----------
    pos1, pos2 : float
        Positions of the two particles (Å).
    eps, sigma : float
        LJ parameters.
    L : float
        Box length (Å).

    Returns
    -------
    V : float
        Potential energy (eV).
    """
    
    D = pos1 - pos2
    
    D -= np.rint( D/L )*L    
    
    V = 4*eps* ((sigma/D)**12   - (sigma/D)**6 )
   
    return V        #  [eV]          



# ----------------------------
# 4. Force calculation
# ----------------------------

Fs=[]

def Force(pos1 , pos2, eps, sigma , L):
    """
        Compute the force on each particle due to LJ interaction.

        In 1D, the force is:

            F = -dV/dr = 4ε [ 12 (σ^12 / r^13) - 6 (σ^6 / r^7) ]

        Implemented as:

            F = 4ε [ 12 (σ/r)^12 - 6 (σ/r)^6 ] / r

        The force on particle 1 is F, and on particle 2 is -F.

        Returns
        -------
        F1, F2 : float
            Forces on particle 1 and 2, respectively (eV/Å).
    """

    # Displacement between particles
    D = pos1 - pos2

    # Apply periodic boundary conditions (minimum image)
    D -= np.rint( D/L )*L    

    # Lennard-Jones force expression in 1D  
    F = 4*eps* (  12 * ( sigma/D)**12   - 6 *  (sigma/D)**6 ) / D

    Fs.append(F) # store for analysis/plotting if desired
    F1 = F
    F2 = -F

    return F1, F2   #  [eV /A]                                       




# --------------------------------------------------
# 5. Velocity-Verlet integration (1D, two particles)
# --------------------------------------------------

def verlet_1d(pos1, pos2, vel1, vel2, dt,  eps, sigma, L):
    
    """
    Perform one Velocity-Verlet integration step for the 1D two-particle system.

    Algorithm:
        1) Compute forces F1, F2 at current positions.
        2) Use F/m to get accelerations a1, a2.
        3) Update positions:
               x_new = x + v*dt + 0.5*a*dt^2
           and apply periodic boundaries.
        4) Compute new forces at updated positions.
        5) Update velocities:
               v_new = v + 0.5*(a + a_new)*dt

    Returns
    -------
    pos_new1, pos_new2, vel_new1, vel_new2
    """

    # Forces at current positions
    F1, F2 = Force(pos1 , pos2,eps , sigma, L ) 

    # Accelerations
    a1 = F1/m      
    a2 = F2 /m

    # New positions using current velocities and accelerations
    pos_new1 = pos1 + vel1 * dt + a1* dt**2 / 2    
    pos_new2 = pos2 + vel2 * dt + a2* dt**2 / 2    

    # Apply periodic boundary conditions to positions
    pos_new1 -= np.rint( pos_new1/L )*L    
    pos_new2 -= np.rint( pos_new2/L )*L    

    # Forces at new positions
    Fnew1 , Fnew2 = Force(pos_new1 , pos_new2,eps , sigma, L)
    
    # New accelerations
    a_new1 = Fnew1 / m
    a_new2 = Fnew2 / m

    # New velocities using average of old and new accelerations
    vel_new1  =  vel1 + (a_new1 + a1) * dt / 2      
    vel_new2  =  vel2 + (a_new2 + a2) * dt / 2      
              
    
    return pos_new1, pos_new2, vel_new1, vel_new2



# ---------------------------------------------
# 6. Initial conditions and quick sanity checks
# ---------------------------------------------
pos1 ,pos2 , vel1, vel2 , N = initialization_1d(2)
potential(pos1, pos2, eps, sigma,L)
Force(pos1 , pos2, eps, sigma, L)



# ----------------------------
# 7. Time-stepping setup
# ----------------------------

dt= [2e-16] 
tmax = 5e-12



"""  INITIIALIZATION  """

pos1 ,pos2 , vel1, vel2 , N =initialization_1d(2)

x_t1 = []     # positions of particle 1 over time
x_t2 = []     # positions of particle 2 over time

vel_t1 = []   # velocities of particle 1 over time
vel_t2 = []   # velocities of particle 2 over time

V_t = []      # potential over time

KE_t1 = []    # kinetic energy of particle 1 over time
KE_t2 = []    # kinetic energy of particle 2 over time

H_t = []      # hamiltonian over time


# Initial values (t = 0)
x_t1.append( pos1 ) 
x_t2.append( pos2 )

D = pos1-pos2
V_t.append( 4*eps* ((sigma/D)**12   - (sigma/D)**6 ) )

vel_t1.append( vel1 )
vel_t2.append( vel2 )

KE_t1.append( kin_en(vel1 , vel2, m)[0] )      # append only the KE1 value that the kin_en function returns
KE_t2.append( kin_en(vel1 , vel2 , m)[1] )      # append only the KE2 value that the kin_en function returns

H_t.append( KE_t1[0]  + KE_t2[0]  + V_t[0] )    # H = Sum( KE ) + V



# ----------------------------
# 9. Main time integration loop
# ----------------------------
        
for h in dt:

    # Number of integration steps for this time step
    NumOfSteps = int(np.rint(tmax / h))

    for i in range( 1, NumOfSteps):

        pos_new1, pos_new2, vel_new1, vel_new2 = verlet_1d( pos1, pos2, vel1, vel2, h,  eps, sigma, L)

        
    # Update the lists
        x_t1.append( pos_new1 ) 
        x_t2.append( pos_new2 )

        V_t.append( potential(pos_new1 , pos_new2, eps , sigma, L) )

        vel_t1.append( vel_new1 )
        vel_t2.append( vel_new2 )

        KE_t1.append( kin_en(vel_new1 , vel_new2, m) [0] )      
        KE_t2.append( kin_en(vel_new1 , vel_new2, m) [1] ) 

        H_t.append( KE_t1[-1]  + KE_t2[-1]  + V_t[-1] )    # H = Sum( KE ) + V


        pos1 = pos_new1    
        pos2 = pos_new2
        vel1 = vel_new1
        vel2 = vel_new2



    t_val = np.linspace(0 , tmax, NumOfSteps)
    
    
    # Positions vs time
    plt.figure()
    plt.plot(t_val, x_t1 , label= f'$x_1(t)$ , dt={h}')
    plt.plot(t_val, x_t2 , label= f'$x_2(t)$, dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel(" Position [ \u212B ]")
    plt.legend()
    # plt.savefig(f'x_t12_{h}.png')


    # Velocities vs time
    plt.figure()
    plt.plot(t_val, vel_t1 ,label= f'$v_1(t)$, dt={h}' )
    plt.plot(t_val, vel_t2 ,label= f'$v_2(t)$, dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [ \u212B / s ]")
    plt.legend()
    # plt.savefig(f'u_t12_{h}.png')

    # Potential and KE_1 vs time
    plt.figure()
    plt.plot(t_val, V_t , label = f'$V(t)$ , dt={h}' , linewidth=1.0 )
    plt.plot(t_val, KE_t1 ,label= f'$KE_1(t)$, dt={h}', linewidth=1.0 )
    plt.xlabel("time [s]")
    plt.ylabel(" V , KE [eV]")
    plt.legend()
    # plt.savefig(f'V_KE_{h}.png')


    # plt.figure()
    # plt.plot(t_val, V_t , label = f'$V(t)$ , dt={h}' )
    # plt.xlabel("time [s]")
    # plt.ylabel("Potential [eV]")
    # plt.legend()
    # plt.savefig(f'V_{h}.png')


    # plt.figure()
    # plt.plot(t_val, KE_t1 ,label= f'$KE_1(t)$ , dt={h}'  )
    # plt.plot(t_val, KE_t2 ,label= f'$KE_2(t)$ , dt={h}')
    # plt.xlabel("time [s]")
    # plt.ylabel("Kinetic energy [eV] ")
    # plt.legend()
    # plt.savefig(f'KE_12_{h}.png')


    # Total H, KE_1, and V on same plot
    plt.figure()
    plt.plot(t_val, H_t ,label= f'$H(t) $, dt={h}' , linewidth=1.0 )
    plt.plot(t_val, KE_t1 ,label= f'$KE_1(t)$ , dt={h}', linewidth=1.0)
    plt.plot(t_val, V_t ,label= f'$V(t)$ , dt={h}', linewidth=1.0)
    plt.xlabel("time [s]")
    plt.ylabel(" H , KE , V [eV]")
    plt.legend()
    # plt.savefig(f'H_V_KE_{h}.png')


    # Total energy alone vs time
    plt.figure()
    plt.plot(t_val, H_t , label= f'$H(t) $, dt={h}')
    plt.xlabel("time [s]")
    plt.ylabel(" H [eV]")
    plt.legend()
    # plt.savefig(f'H_{h}.png')



plt.show

