from colt import from_commandline
import os
import numpy as np
from pysurf.database.dbtools import load_database
from pysurf.utils.osutils import exists_and_isfile

@from_commandline("""                  
i_traj = 0 :: int
f_traj = 0 :: int
noinit_w = False :: bool
""")
def Statistics(i_traj, f_traj, noinit_w):
    """
    Average values over trajectories from i_traj to f_traj
    """
    init_w = not noinit_w  # If init_w == True the initial trajectory weights are readjusted
    
    # Get times from the first trajectory
    prop_db = load_database(os.path.join('traj_' + '%8.8i' % i_traj, 'results.db'))
    time    = prop_db['time'][:].data
    ntimes  = len(time)
    
    # Parameters of the dynamics
    nmodes     = prop_db.dimensions['ncoo'].size
    nstates    = prop_db.dimensions['nstates'].size
    sampling   = prop_db['sampling'][0]
    init_state = prop_db['init_state'][0]
    prop_db.close()
    
    # Average the states, coordinates and momenta
    n_traj = f_traj - i_traj + 1    
    crd_ave = np.zeros((ntimes, nmodes))
    vel_ave = np.zeros((ntimes, nmodes))
    
    #
    # DIFFERENT WAYS TO WEIGHT THE DIFFERENT TRAJECTORIES 
    # AND EVALUATE THE POPULATION
    #
    # Get the initial electronic coordinates and momenta
    
    x0 = []
    p0 = []
    id_traj = []
    
    for it in range(n_traj):
        db_file = os.path.join('traj_' + '%8.8i' % (i_traj + it), 'results.db')
        
        # Check that the database exists and can be used
        
        if not exists_and_isfile(db_file):
            print('Skip data from trajectory n. %i' % (i_traj + it))
            continue
        #
        prop_db = load_database(db_file)
        
        if len(prop_db['time'][:].data) < ntimes:
            print('Too short propagation for trajectory n. %i' % (i_traj + it))
            continue
        
        # check for nans
        if np.isnan(prop_db['crd']).any():
            print('nan found for trajectory n. %i' % (i_traj + it))
            continue
        
        #        
        id_traj.append(i_traj + it)
        x0.append(prop_db['elec'][0,:,0].data)
        p0.append(prop_db['elec'][0,:,1].data)
        prop_db.close()
    
    # New number of trajectories
    n_traj = len(x0)
    print('%i trajectories are used for the analysis' % (n_traj))
    
    x0 = np.array(x0)
    p0 = np.array(p0)
        
    #    
    # Projection operators for i_pop:
    
    if sampling == 'seo':
        n_pop = 3
        i_pop = [0, 1, 2]
        n_weight = 1
        weight = np.zeros((n_traj, n_weight))
        i_pop_weight = [0,0,0]
        # Compute the weights associated to the sampling scheme
        scale = (4 / np.sqrt(np.e) - 1) / n_traj
        #
        for it in range(n_traj):
            weight[it,0] = np.sign(x0[it,init_state]**2 + p0[it,init_state]**2 - 0.5) * scale
        if init_w:
            weight[:,0] = adjust_weights_pop(weight[:,0], x0, p0, [0,1], init_state)
    
    #
    if sampling == 'gauss':
        n_pop = 3
        i_pop = [0, 1, 3]
        n_weight = 2
        weight = np.zeros((n_traj, n_weight))
        i_pop_weight = [0, 0, 1]
        # Compute the weights associated with different sampling schemes
        for it in range(n_traj):
            x2p2 = x0[it,init_state]**2 + p0[it,init_state]**2
            R2 = np.dot(x0[it],x0[it]) + np.dot(p0[it],p0[it])
            weight[it,0] = (2 * x2p2 - 1) / n_traj
            weight[it,1] = (0.5 * x2p2 + (1 - 0.5 * R2) / nstates) / n_traj
        if init_w:
            weight[:,0] = adjust_weights_pop(weight[:,0], x0, p0, [0,1], init_state)
            weight[:,1] = adjust_weights_pop(weight[:,1], x0, p0, [3], init_state, norm = 1.0 / nstates)  
            
    #
    if sampling == 'sphere':
        n_pop = 1
        i_pop = [2]
        n_weight = 1
        weight = np.zeros((n_traj, n_weight))
        i_pop_weight = [0]
        # Compute the weights for the spin mapping
        for it in range(n_traj):
            x2p2 = x0[it,init_state]**2 + p0[it,init_state]**2
            R2 = np.dot(x0[it],x0[it]) + np.dot(p0[it],p0[it])
            weight[it,0] = (0.5 * x2p2 + (1 - 0.5 * R2) / nstates) / n_traj * nstates  ## CHECK THIS FACTOR nstates !!!
        if init_w:
            weight[:,0] = adjust_weights_pop(weight[:,0], x0, p0, [2], init_state, normalize = False)
            
    #
    if sampling in ['spin', 'ehrenfest']:
        n_pop = 1
        i_pop = [2]
        n_weight = 1
        weight = np.zeros((n_traj, n_weight))
        i_pop_weight = [0]
        # Compute the weights for the spin mapping
        for it in range(n_traj):
            x2p2 = x0[it,init_state]**2 + p0[it,init_state]**2
            R2 = np.dot(x0[it],x0[it]) + np.dot(p0[it],p0[it])
            scale = R2 * ((nstates - 1 + 2 / R2) / nstates)**nstates - 1  # Check whether this is correct for nstates > 2
            weight[it,0] = np.sign(0.5 * x2p2 + (1 - 0.5 * R2) / nstates) / n_traj * scale
        if init_w:
            weight[:,0] = adjust_weights_pop(weight[:,0], x0, p0, [2], init_state, normalize = False)          
        
    # Average the coordinates, velocities and electronic populations
    pop_ave = np.zeros((ntimes, n_pop, nstates))
    crd_ave = np.zeros((n_weight, ntimes, nmodes))
    vel_ave = np.zeros((n_weight, ntimes, nmodes))
    cis_ave = np.zeros((ntimes, n_pop, nstates))
    tra_ave = np.zeros((ntimes, n_pop, nstates))
    #
    for it in range(n_traj):
        if np.mod(it,100) == 0:
            print('Trajectory n. %i' % id_traj[it])
        prop_db = load_database(os.path.join('traj_' + '%8.8i' % id_traj[it], 'results.db'))
        # Check that the sampling is consistent
        if prop_db['sampling'][0] != sampling:
            print('ERROR: Different sampling scheme found for the trajectory %i' % it)
        
        # Average position and velocity
        crd = prop_db['crd'][:].data
        vel = prop_db['vel'][:].data
        for j in range(n_weight):
           crd_ave[j] += weight[it,j] * crd
           vel_ave[j] += weight[it,j] * vel
           
        # Average electronic and cis/trans populations
        elec = prop_db['elec'][:].data
        cis   = np.array([1 if (np.mod(np.abs(theta), 2 * np.pi) / np.pi > 0.5 and
                                np.mod(np.abs(theta), 2 * np.pi) / np.pi <= 1.5)  
                          else 0 for theta in crd[:,0]])
                         
        for n in range(ntimes):
            pop = mapping_population(elec[n,:,0], elec[n,:,1])
            for j in range(n_pop):
                pop_ave[n,j] += weight[it, i_pop_weight[j]] * pop[i_pop[j]]
                cis_ave[n,j] += weight[it, i_pop_weight[j]] * cis[n] * pop[i_pop[j]]
                tra_ave[n,j] += weight[it, i_pop_weight[j]] * (1 - cis[n]) * pop[i_pop[j]]
        
        
        # Close the database
        prop_db.close()

    #
    # Write the result 
    #
    pop_type = ['LSC-IVR', 'PBME', 'Impr. pop', 'Impr. pop.']
    
    with open('population_diab.out', 'w') as pop:
        pop.write('# Sampling type: %s \n' % sampling)
        pop.write('# Population types: ')
        for j in range(n_pop):
            pop.write('%s  ' % pop_type[i_pop[j]])
        pop.write('\n #\n')
        for n in range(ntimes):
            pop.write('%8.2f ' % time[n])
            for j in range(n_pop):
                pop.write('  ')
                for i in range(nstates):
                    pop.write(' %7.4f' % pop_ave[n,j,i])
            pop.write('\n')
            
    with open('cis_trans_diab.out', 'w') as pop:
        pop.write('# Sampling type: %s \n' % sampling)
        pop.write('# Population types: ')
        for j in range(n_pop):
            pop.write('%s  ' % pop_type[i_pop[j]])
        pop.write('\n #\n')
        for n in range(ntimes):
            pop.write('%8.2f ' % time[n])
            for j in range(n_pop):
                pop.write('    ')
                for i in range(nstates):
                    pop.write(' %7.4f' % cis_ave[n,j,i])
                    pop.write(' %7.4f' % tra_ave[n,j,i])
                    pop.write('  ')
            pop.write('\n')            
    #
    return None



###################################
# Population functions

def mapping_population(x,p):
    """
    Calculate the electronic population in different ways
    """
    R2 = np.dot(x,x) + np.dot(p,p)
    W00 = np.exp(-R2)
    dummy = (1 - 0.5 * R2) / len(x)
    x2p2 = x**2 + p**2
    #
    return [
        2 ** len(x) * W00 * (2 * x2p2 - 1),   # LSC-IVR
        0.5 * (x2p2 - 1),                     # PBME
        dummy + 0.5 * x2p2,                   # unity
        2 * x2p2 - 1                          # Improved population operator
        ]



def adjust_weights_pop(w0, x0, p0, pop_type, init_state, norm = 1.0, normalize = True):
    """
    Re-adjust the trajectory weights such that average electronic population
    is exactly 0 or 1.

    Parameters
    ----------
    w0 : float
        Vector containing the original trajectory weights
    x0, p0 : float (n_traj, nstates)
        Electronic coordinates and momenta
    pop_type : int
        Population operators used to readjust the weghts
    init_state : int
        Initially populated electronic state
    norm : float
        Desired norm 
    normalize : bool
        Whether the norm should be considered as an observable for re-weighting

    Returns
    -------
    A vector with the modified weights

    """
    eps = 0
    #eps = 1e-12   # regularization parameter
    #
    print('\n RE-WEIGHTING \n')
    n_traj, nstates = x0.shape
    # Number of observables: populations for each state plus total population
    n_pop = len(pop_type)
    n_obs = n_pop * nstates + (1 if normalize else 0)
    b = np.zeros(n_obs)
    F = np.zeros((n_traj, n_obs))
    
    # 
    # Setup the b vector, which contains the exact values of the observables
    #
    i_start = 0
    if normalize:
        b[0] = norm  # norm
        i_start = 1
    for i in range(n_pop):
        b[i_start + i * nstates + init_state] = 1.0
        
    #
    # Setup the F matrix which contains the observables for each trajectory
    #
    for it in range(n_traj):
        x = x0[it]
        p = p0[it]
        pop = mapping_population(x, p)
        i_start = 0
        if normalize:
            F[it,0] = 1  # norm
            i_start = 1
        for i in range(n_pop):
            k = i_start + i * nstates
            F[it,k:k + nstates] = pop[pop_type[i]]        
            
    #
    # Compare the calculated observables with the desired values
    #
    b0 = F.T @ w0
    print('  Observable     Approx.    Exact \n')
    for i in range(n_obs):
        print('      %2i        %7.3f   %7.3f' % (i, b0[i], b[i]))
        
    #
    # Calculate the new weights
    #
    w = w0 + F @ np.linalg.inv(F.T @ F + np.eye(n_obs) * eps) @ (b - b0)
    rel_change = np.sqrt(np.dot(w - w0,w - w0) / np.dot(w,w))
    print('Relative change in the weights: %5.2f %%' % (rel_change * 100))
    print('... after the change...')
    b0 = F.T @ w
    print('  Observable     Approx.    Exact \n')
    for i in range(n_obs):
        print('      %2i        %7.3f   %7.3f' % (i, b0[i], b[i]))
    #
    print('\n')
    return w
    


###################################

if __name__ == '__main__':
    _ = Statistics()
    exit()
    
    



        
    