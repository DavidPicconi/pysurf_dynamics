from colt import from_commandline
import numpy as np
import os

# Import the propagators
from pysurf.dynamics.base_propagator import PropagatorFactory
QuasiClassicalMapping = PropagatorFactory._methods['QuasiClassicalMapping']

# PySurf imports
from colt import Colt
from pysurf.utils import exists_and_isfile
from pysurf.utils.constants import fs2au
from pysurf.logger import get_logger
from pysurf.spp import SurfacePointProvider
from pysurf.sampling import DynCondition

from pysurf.system.atominfo import MASSES
from pysurf.constants import U_TO_AMU
from pysurf.utils.constants import fs2au 

### Database tools
from pysurf.database.database import Database
from pysurf.database.dbtools import DBVariable



#####################################################
#
# Driver for the dynamics using mapping approaches
#
#####################################################

class MappingDynamics(QuasiClassicalMapping):
    
    _user_input = """
    
    # Total propagation time in fs
    time_final = :: float                                                            
    
    # Time step in fs for the propagation                                                            
    time_step  = :: float
    
    # Time step for printing the results
    time_output = :: float
    
    # Active states
    states = :: flist_np
    
    # Initial state
    init_state = :: int
    
    # Representation for the initial state
    state_repr = :: str
    
    # Initial coordinates
    Q = :: flist_np
    
    # Initial momenta
    P = :: flist_np
    
    # Random seed
    random_seed = :: int
    
    """
    
    @classmethod
    def from_config(cls, config, config_dyn):
        return cls(config, config_dyn)
    
    @classmethod
    def from_inputfile(cls, inputfile, config_dyn):
        # Generate the config
        if exists_and_isfile(inputfile):
            config = cls.generate_input(inputfile, config = inputfile,
                                        ask_defaults = False)
        else:
            config = cls.generate_input(inputfile, ask_defaults = False)
        #
        return cls.from_config(config, config_dyn)
    
    
    def __init__(self, config, config_dyn):
        
        method = config_dyn['method'].lower()
        
        #
        # Initialize the surface-point provider         #
        #
        
        self.spp    = config_dyn['spp']
        model  = config_dyn['model']
        masses = config_dyn['masses']
        self.masses = masses
        
        self.nstates = len(config['states'])
        np.random.seed(config['random_seed'])
        
        self.state_repr = config['state_repr']
        if self.state_repr == 'diabatic' : self.properties = ['Hel', 'Hel_gradient']
        if self.state_repr == 'adiabatic': self.properties = ['energy', 'gradient', 'nacs']
        
        ############################
        # For ab initio models, reshape the coordinate/moments
        
        if model == 0:
            crd = config['Q'].reshape((len(masses), 3))
            vel = config['P'].reshape((len(masses), 3)) / np.c_[masses,masses,masses]
            
        if model == 1:
            crd = config['Q']
            vel = config['P'] / masses
        
        #
        self.init = DynCondition(crd = crd, veloc = vel, state = 0)
        
        
        ############################
        #
        # Initial for the electronic variables
        #
        
        self.x = np.zeros(self.nstates)
        self.p = np.zeros(self.nstates)
        
        #
        # Sampling of the electronic coordinates
        #
            
        # Import the samplers for the mapping variables
        from pysurf.sampling.base_sampler import SamplerFactory
        SchwingerSampling = SamplerFactory._methods['Schwinger']
        SpinSampling      = SamplerFactory._methods['SpinSampler']
        
        # Choose the sampling method
        _sampling = {'lsc-ivr': 'seo',
                     'pbme': 'seo',
                     'unity': 'gauss',
                     'spin': 'spin',
                     'ehrenfest': 'ehrenfest'}
        
        sampling = _sampling[method]

        # Sample        
        
        if sampling.lower() not in ['gauss', 'seo', 'sphere', 'spin', 'ehrenfest']:
            print('ERROR: The sampling scheme must be: gauss, seo, sphere or spin')
            exit()
        
        
        if sampling.lower() == 'gauss':
            # Sample from the distribution exp(-x^2-p^2) 
            self.x = np.random.normal(loc = 0.0, scale = np.sqrt(0.5),
                                      size = self.nstates)
            self.p = np.random.normal(loc = 0.0, scale = np.sqrt(0.5),
                                      size = self.nstates)
        
        
        if sampling.lower() == 'seo':
            # Sample the variables for the populated state 
            # from the distribution exp(-x^2-p^2) * (2 * x^2 + 2 * p^2 - 1)
            # and the variables for the other states from exp(-x^2-p^2)
            S_samp = SchwingerSampling()
            for i in range(self.nstates):
                if config['init_state'] == i:
                    cond = S_samp.get_condition()
                    self.x[i] = cond.crd[0]
                    self.p[i] = cond.veloc[0]
                else:
                    self.x[i] = np.random.normal(loc = 0.0, scale = np.sqrt(0.5))
                    self.p[i] = np.random.normal(loc = 0.0, scale = np.sqrt(0.5))
        
        
        if sampling.lower() == 'sphere':
           # Sample from a sphere 
           # Sum_(i = 1)^N  (x_i^2 + p_i^2) = 2 * sqrt(N + 1)
           # (spin-mapping)

           self.x = np.random.normal(loc = 0.0, scale = np.sqrt(0.5), size = self.nstates)
           self.p = np.random.normal(loc = 0.0, scale = np.sqrt(0.5), size = self.nstates)
           
           R2 = np.dot(self.x, self.x) + np.dot(self.p, self.p)
           scale = np.sqrt(2 * np.sqrt(self.nstates + 1) / R2)
           self.x *= scale
           self.x *= scale
        
        
        if sampling.lower() == 'spin':
            # Sampling for the spin mapping based on spherical coordinates
            
            iS = config['init_state']
            R  = np.sqrt(2 * np.sqrt(self.nstates + 1))
            
            S_samp = SpinSampling()
            alpha = S_samp.get_condition(self.nstates)
            
            # state indices in the order of filling
            ind = np.arange(self.nstates)
            ind[0]  = iS
            ind[iS] = 0
            
            # Set up the Rm variables: Rm = (x_m^2 + p_m^2)^(1/2)
            Rm = np.zeros(self.nstates)
            Rm[0] = R * np.cos(alpha)
            dummy = R * np.sin(alpha)
            for i in range(2,self.nstates):
                alpha = np.arcsin(np.random.rand() ** (1 / (2 * self.nstates - 2 * i)))
                Rm[i - 1] = dummy * np.cos(alpha)
                dummy *= np.sin(alpha)
            Rm[self.nstates - 1] = dummy
            
            for i in range(self.nstates):
                phi = np.random.rand() * 2 * np.pi
                self.x[ind[i]] = Rm[i] * np.cos(phi)
                self.p[ind[i]] = Rm[i] * np.sin(phi)
                
        if sampling.lower() == 'ehrenfest':
            # Sampling for Ehrenfest dynamics using the mapping variables
            
            iS = config['init_state']
            phi = np.random.rand() * 2 * np.pi
            self.x[iS] = np.sqrt(2.0) * np.cos(phi)
            self.p[iS] = np.sqrt(2.0) * np.sin(phi)
        
        
        ############################
        #
        # Times and other attributes
        #
        
        self.time_final = config['time_final']  * fs2au
        self.time_step  = config['time_step']   * fs2au
        self.tout       = config['time_output'] * fs2au
        self.restart    = False
                
        ############################
        #
        # Initialise the database
        #
        
        settings = {
                    'dimensions': {
                    'frame'  : 'unlimited',
                    'ncoo'   : len(self.masses),
                    'nstates': self.nstates,
                    'one'    : 1,
                    'two'    : 2,
                    'three'  : 3
                       },                         
                    'variables': {
                    'crd'       : DBVariable(np.double, ('frame', 'ncoo')) if model == 1 else DBVariable(np.double, ('frame', 'ncoo', 'three')),
                    'vel'       : DBVariable(np.double, ('frame', 'ncoo')) if model == 1 else DBVariable(np.double, ('frame', 'ncoo', 'three')),
                    'elec'      : DBVariable(np.double, ('frame', 'nstates', 'two')),  # mapping variables
                    'time'      : DBVariable(np.double, ('frame', 'one')),
                    'Hel'       : DBVariable(np.double, ('frame', 'nstates', 'nstates')),
                    'sampling'  : DBVariable(str, ('one')),
                    'init_state': DBVariable(int, ('one'))                        
                      }
                    }
        
        self.db = Database('results.db', settings)
        self.db.set('sampling', np.array([sampling]))
        self.db.set('init_state', np.array([config['init_state']]))
        
        # Run the propagation
        #
        
        self._run(int(self.time_final / self.time_step),
                  self.time_step)
     
    #    
    def output_step(self, istep, time, ekin, elec, etot):
        pass




#####################################################
#
# Driver for the dynamics using surface hopping approaches
#
#####################################################

class SurfaceHoppingDynamics(Colt):
    
    _user_input = """
    
    # Total propagation time in fs
    time_final = :: float                                                            
    
    # Time step in fs for the propagation                                                            
    time_step  = :: float
    
    # Time step for printing the results
    time_output = :: float
    
    # Active states
    states = :: ilist_np
    
    # Initial state
    init_state = :: int
    
    # Representation for the initial state
    state_repr = :: str
    
    # Initial coordinates
    Q = :: flist_np
    
    # Initial momenta
    P = :: flist_np
    
    # Random seed
    random_seed = :: int
    
    """
        
    
    @classmethod
    def from_config(cls, config, config_dyn):
        return cls(config, config_dyn)
    
    @classmethod
    def from_inputfile(cls, inputfile, config_dyn):
        # Generate the config
        if exists_and_isfile(inputfile):
            config = cls.generate_input(inputfile, config = inputfile,
                                        ask_defaults = False)
        else:
            config = cls.generate_input(inputfile, ask_defaults = False)
        #
        return cls.from_config(config, config_dyn)
    
    
    def __init__(self, config, config_dyn):
        
        #self.logger = get_logger('propagation.log', 'propagator')
        
        method = config_dyn['method'].lower()
        
        #
        # Initialize the surface-point provider         #
        #
        
        spp    = config_dyn['spp']
        model  = config_dyn['model']
        masses = config_dyn['masses']
       
        nstates = len(config['states'])
        np.random.seed(config['random_seed'])
        
        ############################
        # For ab initio models, reshape the coordinate/moments
        
        if model == 0:
            crd = config['Q'].reshape((len(masses), 3))
            vel = config['P'].reshape((len(masses), 3)) / np.c_[masses,masses,masses]
            
        if model == 1:
            crd = config['Q']
            vel = config['P'] / masses
            
        ############################
        #
        # Initialize the initially active state and the coefficients
        #
        
        i0 = config['init_state']
        
        el_repr = config['state_repr']
        #
        ## check: the diabatic representation can only be used with models
        if el_repr == 'diabatic' and model == 0:
            print('ERROR: At present the diabatic representation ')
            print('       can only be used with models')
            exit()
        #
        coeff = np.zeros(nstates, dtype = np.complex128)
        
        # FSSH
        if method in ['fssh', 'lzsh']:
            coeff[i0] = np.exp(np.random.rand() * 2 * np.pi * 1j)
            
            if el_repr == 'diabatic':
                result = spp.request(crd, ['Hel'])
                ene, C = np.linalg.eigh(result['Hel'])
                #ene, C = np.linalg.eigh(spp._diab_Hel(config['Q']))
                coeff  = C.T @ coeff
                i0     = np.random.choice(np.arange(nstates),
                                          p = (coeff.conj() * coeff).real )
                
        # MISH 
        if method == 'mish':
            # Initialize the coefficients according to Runeson & Manolopoulos, JCP 159, 094115 (2023)
            
            accepted = False
            while not accepted:
                coeff_0 = np.random.normal(size = (nstates, 2))
                if np.argmax(np.sum(coeff_0**2, axis = 1)) == i0:
                    accepted = True
                    
            # normalize
            dummy = np.sum(coeff_0**2)
            coeff_0 /= np.sqrt(dummy)
            #
            coeff = np.array([c[0] + 1j * c[1] for c in coeff_0])
            
            if el_repr == 'diabatic':
                result = spp.request(crd, ['Hel'])
                ene, C = np.linalg.eigh(result['Hel'])
                coeff = C.T @ coeff
                i0 = np.argmax((coeff.conj() * coeff).real)
        

        
        ############################
        #
        # Launch
        #
        
        from pysurf import user_plugins
        from sys import path as sys_path
        sys_path.append(user_plugins)
        from vv_three_deco_fssh_lzsh_propagator import VelocityVerletPropagator, State
        
        _prob = {'fssh': 'tully',
                 'lzsh': 'lz',
                 'mish': 'mish'}
        
        _config = {}
        _config['rescale_vel'] = 'momentum' if method == 'lzsh' else 'nacs'
        _config['substeps'] = 'false'
        _config['thermostat'] = 'false'
    
        
        state = State(
            crd     = crd,
            vel     = vel,
            mass    = masses,
            model   = model,
            t       = 0.0,
            dt      = config['time_step'] * fs2au,
            #tout    = config['time_output'],
            mdsteps = int(config['time_final'] / config['time_step']),
            instate = i0 ,
            nstates = nstates,
            states  = config['states'],
            ncoeff  = coeff.copy(),
            prob    = _prob[method.lower()],
            #
            rescale_vel = 'momentum' if method == 'lzsh' else 'nacs',
            coupling = 'non_coup' if method == 'lzsh' else 'nacs',
            method = 'Surface_Hopping' if nstates > 1 else 'Born_Oppenheimer',
            decoherence = 'EDC' if method == 'fssh' else 'No_DC',
            thermostat = False,
            #
            substeps = False,
            config = _config,                 # Use to pass information about the thermostat and the substeps
            atomids = config_dyn['atomids']   # This 
            )
        
        state.tout = config['time_output'] * fs2au
               
        prop = VelocityVerletPropagator(state, spp)
        prop.run()
        
        #
        
        return







#####################################################################################

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


@from_commandline("""
# Index of the first trajectory (default is 0)  
i_traj = 0 :: int

# Index of the last trajectory (default is 0)
f_traj = 0 :: int

# Method for quantum-classical dynamics.
# Available methods are:
#    fssh, lzsh, mash, lsc-ivr, pbme, unity, spin
# Default is fssh.
method = fssh :: str 

# Surface-point-provider file
spp_data = spp.inp :: str

# Dynamics setup file
dyna_file = dyna.inp :: str
""")
def run_dynamics(i_traj, f_traj, method, spp_data, dyna_file):
    """ 
    Launch sequential simulation of mapping trajectories
    """
    
    #
    # Check that the method is implemented
    #
    
    _methods_sh  = ["fssh", "lzsh", "mish"]              # surface-hopping methods
    _methods_map = ["lsc-ivr", "pbme", "unity", "spin", "ehrenfest"]  # mapping ("Ehrenfest-like") methods
    
    _methods = _methods_sh + _methods_map
    
    if method.lower() not in _methods:
        print(f'The method {method} is not implemented.')
        print('Please use one of the following methods:')
        print(' %s' % str(_methods).replace('[','').replace("'",'').replace(']',''))
        #
        return None
    
    #
    # Set up the surface point provider
    #
    
    info = System_Information.from_inputfile(dyna_file)
    
    spp = SurfacePointProvider.from_questions(config     = spp_data,
                                              properties = [],
                                              nstates    = info.n_states,
                                              natoms     = info.n_part,
                                              atomids    = info.atomids
                                              )
    
    #
    # Configure the dynamics
    #
    
    config_dyn = {'method': method.lower(),
                  'spp'   : spp}
    
    config_dyn['model'] = 0 if spp._interface.__class__.__name__ in SurfacePointProvider._modes['ab-initio'].__dict__['software'].keys() else 1
    
    if config_dyn['model'] == 1:
        config_dyn['masses'] = spp._interface.masses.copy()
    else:
        config_dyn['masses'] =  np.array([MASSES[i] * U_TO_AMU for i in info.atomids])
        
    config_dyn['atomids'] = info.atomids
    
    #############################
    n_traj = f_traj - i_traj + 1
    
    cwd = os.getcwd()
    for it in range(n_traj):
        # Set random seed to ensure repeatibility of the results
        #np.random.seed(random_seed + i_traj + it)
        #
        traj_dir = os.path.join(cwd, 'traj_' + '%8.8i'  % (i_traj + it))
        os.chdir(traj_dir)
        print('Running trajectory n. %8.8i... ' % (i_traj + it))
        
        if method.lower() in _methods_map:
            MappingDynamics.from_inputfile('run.inp', 
                                           config_dyn = config_dyn)
            
        if method.lower() in ['fssh', 'mish', 'lzsh']:
            SurfaceHoppingDynamics.from_inputfile('run.inp',
                                                  config_dyn = config_dyn)
            
            
            
        #MappingDynamics.from_inputfile('run.inp', sampling = sampling.lower())
        print('... DONE \n')
    

##################################    

if __name__=="__main__":
    run_dynamics()
    