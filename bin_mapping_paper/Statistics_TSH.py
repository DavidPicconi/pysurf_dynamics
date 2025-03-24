from colt import from_commandline, Colt
import os
import numpy as np
from pysurf.database.dbtools import load_database
from pysurf.utils.constants import fs2au
from pysurf.utils.osutils import exists_and_isfile

from pysurf.spp import SurfacePointProvider
from pysurf.spp.spp import ModelFactory


@from_commandline("""                  
i_traj = 0 :: int
f_traj = 0 :: int
diabatic = False :: bool
""")
def Statistics(i_traj, f_traj, diabatic):
    """
    Average values over trajectories from i_traj to f_traj
    """
    # If the adiabatic/diabatic analysis is required, construct the model
    if diabatic:

        info = System_Information.from_inputfile('dyna.inp')

        spp = SurfacePointProvider.from_questions(config     = 'spp.inp',
                                                  properties = [],
                                                  nstates    = info.n_states,
                                                  natoms     = info.n_part,
                                                  atomids    = info.atomids
                                                 )

        spp = spp._interface

        #spp = get_model(os.path.join('traj_' + '%8.8i' % i_traj, 'run.inp'),
        #                'traj_' + '%8.8i' % i_traj)
                  
    #
    prop_db = load_database(os.path.join('traj_' + '%8.8i' % i_traj, 'results.db'))
    time    = prop_db['time'][:].data
    ntimes  = len(time)   
    # Parameters of the dynamics
    nmodes  = prop_db.dimensions['nmodes'].size
    nstates = prop_db.dimensions['nstates'].size
    prop_db.close()
    # Average the states, coordinates and momenta
    n_traj = f_traj - i_traj + 1    
    crd_ave = np.zeros((ntimes, nmodes))
    vel_ave = np.zeros((ntimes, nmodes))
    
    #
    # Average the electronic populations over the different trajectories
    # Three populations are calculated:
    # 1 - Adiabatic population, calculated from the active state
    # 2 - Adiabatic population, calculated from the coefficients
    # 3 - Diabatic population, calculated from the active state
    # 4 - Diabatic population, calculated from the coefficients
    #
    pop_ave_1 = np.zeros((ntimes, nstates))
    pop_ave_2 = np.zeros((ntimes, nstates))
    if diabatic:
        pop_ave_3 = np.zeros((ntimes, nstates))
        pop_ave_4 = np.zeros((ntimes, nstates))
    
    n_traj_completed = 0
    for it in range(n_traj):
        if np.mod(it, 100) == 0:
            print('Trajectory n. %i' % it)
            
        db_file = os.path.join('traj_' + '%8.8i' % it, 'results.db')
            
        # Check that the database exists and can be used
         
        if not exists_and_isfile(db_file):
            print('Skip data from trajectory n. %i' % (i_traj + it))
            continue
        
        prop_db = load_database(db_file)
        
        if 'time' not in prop_db.variables:
            print('Too short propagation for trajectory n. %i' % (i_traj + it))
            prop_db.close()
            continue
        
        if len(prop_db['time'][:].data) < ntimes:
            print('Too short propagation for trajectory n. %i' % (i_traj + it))
            prop_db.close()
            continue
        
        n_traj_completed += 1
        
        pop   = prop_db['currstate'][:].data
        coeff = prop_db['coeff'][:].data
        #
        for j in range(ntimes):
            coeff[j] = coeff[j].reshape((-1,1)).reshape((2,nstates)).T
        #
        crd   = prop_db['crd'][:].data
                
        # Population "1"
        for j in range(nstates):
            pop_ave_1[:,j] += np.array([1 if int(p) == j else 0 for p in pop])
            
        # Population "2"
        for j in range(nstates):
            pop_ave_2[:,j] += np.array([c[j,0]**2 + c[j,1]**2 for c in coeff])
            
        # Populations "3" and "4"
        if not diabatic:
            prop_db.close()
            continue
        
        for n in range(ntimes):
            # Orthogonal transformation matrix
            
            #if 'time' in spp.implemented:
            #    spp.time = time[n]
                
            Hel = spp._diab_Hel(crd[n])
            U   = np.linalg.eigh(Hel)[1]
            #
            c = np.array([coeff[n,j,0] + 1j * coeff[n,j,1] for j in range(nstates)])
            rho = np.outer(c, c.conj())
            rho_diab = U @ rho @ U.T
            
            for j in range(nstates):
                pop_ave_4[n,j] += rho_diab[j,j].real
            
            #
            rho = np.zeros((nstates,nstates))
            iactive = int(pop[n])
            rho[iactive,iactive] = 1.0
            rho_diab = U @ rho @ U.T
            
            for j in range(nstates):
                pop_ave_3[n,j] += rho_diab[j,j].real
        
        #
        prop_db.close()
    #
    print('%i/%i trajectories completed' % (n_traj_completed, n_traj))
    #
    pop_ave_1 /= n_traj_completed
    pop_ave_2 /= n_traj_completed
    if diabatic:
        pop_ave_3 /= n_traj_completed
        pop_ave_4 /= n_traj_completed
    with open('population.out', 'w') as pop:
        pop.write('Adiabatic pop. (act. state) | Adiabatic pop. (coefficients) | Diabatic pop. (act. state) | Diabatic pop. (coefficient) \n')
        for n in range(ntimes):            
            pop.write(f'{(time[n,0] / fs2au):8.2f}  ')
            for j in range(nstates):
                pop.write(f' {pop_ave_1[n,j]:7.4f} ')
            pop.write('  ')
            for j in range(nstates):
                pop.write(f' {pop_ave_2[n,j]:7.4f} ')
            if diabatic:
                pop.write('  ')
                for j in range(nstates):
                    pop.write(f' {pop_ave_3[n,j]:7.4f} ')
                pop.write('  ')
                for j in range(nstates):
                    pop.write(f' {pop_ave_4[n,j]:7.4f} ')
            pop.write('\n')
        
        

###################################

class System_Information(Colt):

    _user_input = """

    n_states = :: int

    n_part = :: int

    atomids = :: ilist

    """

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @classmethod
    def from_inputfile(cls, inputfile):
        qst = cls.generate_user_input(config = inputfile)
        config = qst.get_answers()
        return cls.from_config(config)

    def __init__(self, config):
        self.n_states = config['n_states']
        self.n_part   = config['n_part']
        self.atomids  = config['atomids']


###
        
        
    

###################################

if __name__ == '__main__':
    _ = Statistics()
    exit()
    
    
###################################
