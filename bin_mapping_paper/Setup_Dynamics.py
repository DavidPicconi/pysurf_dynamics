from colt import from_commandline

from numpy.random import seed, randint
from numpy import array, diag

from pysurf.setup import SetupBase
from pysurf.utils import exists_and_isfile
from pysurf.logger import get_logger
from pysurf.sampling.base_sampler import SamplerFactory
from pysurf.spp.spp import SurfacePointProvider

import os

################################


class Setup_Dynamics(SetupBase):
    
    folder = "."
    subfolder = "traj"
    
    _user_input = """
    
    # Number of trajectories?
    n_traj = :: int
    
    # Total propagation time in fs?
    time_final = :: float                                                            
    
    # Time step in fs for the integration?
    time_step  = :: float
    
    # Time step for printing the results?
    time_output = :: float
    
    # Sampler?
    sampler = :: str
    
    # Surface Point Provider specifications?
    spp_data = :: str
    
    # Number of states to be computed?
    n_states = :: int
    
    # Active states?
    states = :: ilist
    
    # Initally populated state?
    init_state = :: int
    
    # Electronic representation?
    state_repr = :: str :: ["adiabatic", "diabatic"]
    
    # Seed for the random number generator?
    random_state = 1234 :: int
    
    # Number of classical particles?
    n_part = 0 :: int
    
    # Atoms?
    atomids = [] :: ilist
    
    """
    
    _sampling_methods = SamplerFactory._methods
    
    ####
    
    @classmethod
    def from_config(cls, config, logger = None, questions = None, inputfile = 'dyna.inp'):
        return cls(config, logger = logger, questions = questions, inputfile = inputfile)
    
    
    @classmethod
    def from_inputfile(cls, inputfile):
        # Generate the config
        if exists_and_isfile(inputfile):
            qst = cls.generate_user_input(config = inputfile, presets = None)
            config = qst.generate_input(inputfile, config = inputfile,
                                        ask_defaults = False)
        else:
            qst = cls.generate_user_input(config = None, presets = None)
            config = qst.generate_input(inputfile, ask_defaults = False)
        #
        return cls.from_config(config, questions = qst, inputfile = inputfile)
    
    
    @classmethod
    def _extend_user_input(cls, questions):
        questions.generate_cases("sampler", {name: val.colt_user_input for name, val in cls._sampling_methods.items()})
    
    
    #    
    def __init__(self, config, logger = None, questions = None, inputfile = 'dyna.inp'):
        """
        Setup trajectory calculations
        """
        
        # Initialize the logger        
        if logger is None:
            self.logger = get_logger("dyna.log", "setup_dynamics")
            self.logger.header("SETUP OF DYNAMICAL SIMULATIONS", config)
        else:
            self.logger = logger
            
        # Initialize the setup class
        SetupBase.__init__(self, logger)
    
        
        # Setup the random variable
        random_state = config["random_state"]
        seed(random_state)
    
    
        # Get the number of trajectories
        n_traj = config["n_traj"]
        
        
        ###
        # Sampling
        
        sampler_name, sampler_config = config["sampler"].value, config["sampler"].subquestion_answers
        sampler = self._sampling_methods[sampler_name].from_config(sampler_config)
        
        # Get the number of particles
        if "nmodes" in sampler.system.__dict__:
            n_part  = sampler.system.__dict__["nmodes"]
            atomids = array([])
        if "natoms" in sampler.system.__dict__:
            n_part  = sampler.system.__dict__["natoms"]
            atomids = sampler.system.__dict__["atomids"]
         
        # Update the input file, so that the number of particles and the atomids are stored
        questions.set_answer('n_part', n_part)
        questions.set_answer('atomids', str(atomids))
        questions.write_config(inputfile)
            
        # Sample
        initial_conditions = [sampler.get_condition() for _ in range(n_traj)]
            
        
        ###
        # Surface point provider
        states   = config["states"]
        spp_data = config["spp_data"]
        
        ## it's not necessary to generate the spp
        spp = SurfacePointProvider.from_questions(config     = spp_data,
                                                  properties = [],
                                                  nstates    = len(states),
                                                  natoms     = n_part,
                                                  atomids    = atomids
                                                  )
        # nstates, natoms, atomids should be written in the dyna.inp file!
        
        ###
        #
        # CHECKS
        #
        
        # Check that the initial state is among the possible states
        if config["init_state"] not in states:
            print("ERROR \n")
            print("The initial state is not in the list of states")
            exit()
                
        # Check that the output time is a multiple of the time step
        dt   = config["time_step"]
        tout = config["time_output"]
        if abs(round(tout / dt) - tout / dt) > 1e-8:
            print("ERROR \n")
            print("The output time must be a multiple of the time step")
            exit()
            

        ###
        # Create the folders
        self.setup_folders(range(n_traj), config, initial_conditions, sampler.system.masses)
        
        # ######
        # print(config["sampler"])
        # print(sampler_name)
        # print(sampler_config)
        
        
        # print(sampler)
        # print(sampler.get_condition())  # CHECK, because it might sample in mass-weighted coordinates
        #                                 # CHECK the sampling... Where is the temperature used?
                                        
        # print()
        # print(spp)
        
        
    #####
    # Method to initialize the folders (one per trajectory)
    def setup_folder(self, number, foldername, config, initial_conditions, masses):
        #
        with open(os.path.join(foldername, "run.inp"), 'w') as runinfo:
            
            # Propagation details
            runinfo.write("time_final = %8.2f \n"   % config["time_final"])
            runinfo.write("time_step = %12.4f \n"    % config["time_step"])
            runinfo.write("time_output = %12.4f \n" % config["time_output"])
            
            # Info about the spp
            #runinfo.write("spp_data = %s \n" % os.path.join(os.pardir,config["spp_data"]))   # the propagation launcher should move to the parent directory to launch the dynamics
            states = config["states"]
            runinfo.write(f"states = {states} \n")
            runinfo.write("init_state = %i \n" % config["init_state"])
            runinfo.write("state_repr = %s \n" % config["state_repr"])
            
            # Initial coordinates
            cond = initial_conditions[number]
            Q0 = cond.crd
            runinfo.write("Q = " + str(Q0.tolist()) + "\n")
            P0 = (cond.veloc.T @ diag(masses)).T  # this gives the momenta
            runinfo.write("P = " + str(P0.tolist()) + "\n")
                                    
            # Generate a seed 
            runinfo.write("random_seed = %i \n" % randint(0, 10000))



################################

@from_commandline("""
inputfile = dyna.inp :: file 
""")
def setup_dynamics(inputfile):
    """ Setting up initial conditions according to the inputfile.
    If inputfile doesn't exist, colt will ask all necessary questions
    """
    Setup_Dynamics.from_inputfile(inputfile)


if __name__== "__main__":
    setup_dynamics() 