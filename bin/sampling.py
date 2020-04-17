import os
import numpy as np
#
from pysurf.colt import Colt
from pysurf.colt import FromCommandline
#
from pysurf.utils.constants import fs2au
from pysurf.sampling.sampling import Sampling


@FromCommandline("""
inputfile = sampling.inp :: file 
""")
def command_setup_sampling(inputfile):
    """ Setting up initial conditions according to the inputfile.
    If inputfile doesn't exist, colt will ask all necessary questions
    """
    sampling = Sampling.from_inputfile(inputfile)
    sampling = Sampling.from_db('sampling.db')
    print(sampling.get_condition(30))

if __name__=="__main__":
    command_setup_sampling()