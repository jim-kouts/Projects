# ============================================================
#     1D two-particle bouncing (MD) with energy monitoring
# ============================================================


import numpy as np
import math
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 50



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