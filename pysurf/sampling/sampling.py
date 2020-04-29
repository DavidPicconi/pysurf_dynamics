import numpy as np

from .sampling_db import SamplingDB
from .base_sampler import SamplerFactory

from pysurf.utils import exists_and_isfile
from pysurf.logger import Logger, get_logger
from pysurf.colt import Colt
from pysurf.molecule import Molecule


class Sampling(Colt):
    """ Sampling is the header class for the sampling routines. It asks the main questions, selects
        the sampler and reads and writes the conditions to the sampling database.
    """
    _questions = """
    # Number of Calculations
    n_conditions = 100 :: int

    method = :: str

    # Database containing all the initial conditions
    sampling_db = sampling.db :: file

    #State whether the system is a model system
    model = False :: bool
    """

    @classmethod
    def _generate_subquestions(cls, questions):
        questions.generate_cases("method", {name: method.questions
                                 for name, method in SamplerFactory._methods.items()})

    @classmethod
    def from_config(cls, config, logger=None):
        if exists_and_isfile(config['sampling_db']):
            logger.info(f"Found existing database {config['sampling_db']}")
            db = SamplingDB.from_db(config['sampling_db'])
            logger.info(f"There are already {db.nconditions} conditions in the database")
            sampler = None
        else:
            logger.info(f"Loading sampler {config['method'].value}")
            sampler = cls._get_sampler(config['method'])
            logger.info(f"Creating new database {config['sampling_db']}")
            db = SamplingDB.from_sampler(config, sampler)
        return cls(config, db, db.dynsampling, sampler=sampler, logger=logger)

    @classmethod
    def from_inputfile(cls, inputfile):
        logger = get_logger('sampling.log', 'sampling')
        # Generate the config
        quests = cls.generate_questions(config=inputfile)
        config = quests.ask(inputfile)
        logger.header('SAMPLING', config)
        logger.info(f"Took information from inputfile {inputfile}")
        return cls.from_config(config, logger=logger)    

    @classmethod 
    def create_db(cls, dbfilename, variables, dimensions, molecule, modes, model=False, sp=False, logger=None):
        db = SamplingDB.create_db(dbfilename, variables, dimensions=dimensions, molecule=molecule, modes=modes, model=model, sp=sp)
        config = db.get_config()
        config['sampling_db'] = dbfilename
        return cls(config, db, db.dynsampling, logger=logger)

    @classmethod
    def from_db(cls, dbfilename, logger=None):
        db = SamplingDB.from_db(dbfilename)
        config = db.get_config()
        config['sampling_db'] = dbfilename
        return cls(config, db, db.dynsampling, logger=logger)
    
    def __init__(self, config, db, dynsampling, sampler=None, logger=None):
        """ Sampling always goes with a database, if not needed use Sampler class
        """
        self._db = db    
        if logger is None:
            self.logger = get_logger('sampling.log', 'sampling')
            self.logger.header('SAMPLING', config)
        else:
            self.logger = logger

        self.sampler = sampler


        # check for conditions
        if self.nconditions < config['n_conditions']:
            # setup sampler
            if sampler is None:
                logger.info(f"Loading sampler {config['method'].value}")
                self.sampler = self._get_sampler(config['method'])
            else:
                self.sampler = sampler
            self.logger.info(f"Adding {config['n_conditions']-self.nconditions} additional entries to the database")
            self.add_conditions(config['n_conditions'] - self.nconditions)

    def add_conditions(self, nconditions, state=0):
        # TODO
        # Take the random seed and random counter from the database to
        # assure the consistency with all
        # the previous conditions
        for _ in range(nconditions):
            cond = self.sampler.get_condition()
            self._db.append_condition(cond)
    
    def write_condition(self, condition, idx):
        self._db.write_condition(condition, idx)

    def set(self, key, value, idx=None):
        self._db.set(key, value, idx)

    def __iter__(self):
        self._start = 1  # skip equilibrium structure
        return self

    def __next__(self):
        cond = self.get_condition(self._start)
        if cond is not None:
            self._start += 1
            return cond
        raise StopIteration

    def get_condition(self, idx):
        return self._db.get_condition(idx)

    @staticmethod
    def _get_sampler(config):
        return SamplerFactory._methods[config.value].from_config(config)
   
    @property
    def dynsampling(self):
        return self._db.dynsampling
    
    @property
    def info(self):
        return self._db.info

    @property
    def molecule(self):
        self._molecule = Molecule(np.copy(self._db['atomids']),
                                      np.copy(self._db['crd_equi']),
                                      np.copy(self._db['masses']))
        return self._molecule

    @property
    def natoms(self):
        return self.molecule.natoms

    @property
    def atomids(self):
        return self.molecule.atomids

    @property
    def method(self):
        return self._db['method']

    @property
    def modes(self):
        return self._db.modes

    @property
    def model(self):
        return self._db.model

    @property
    def nconditions(self):
        return len(self._db['crd'])

    @property
    def masses(self):
        return self._db.masses
