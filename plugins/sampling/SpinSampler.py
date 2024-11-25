import numpy as np
from pysurf.sampling.base_sampler import DynSamplerBase

class SpinSampler(DynSamplerBase):
    """
    Sample the alpha variable according to the distribution
       G(sin(alpha)) = Abs(((N - 2) * R^2 + 4) / N + R^2 * cos(2 * alpha)) * sin(2 * alpha)
    with R^2 = 2 * sqrt(N + 1) and alpha in the range [0, pi/2]
    """
    @classmethod
    def from_config(cls, config, start=None):
        pass
    
    def get_condition(self, N = 2):
        # Define the maximum of the distribution
        if N == 2:
            Gmax = (1 + np.sqrt(3)) * np.sqrt(2 * (3 + np.sqrt(3))) / 9
        else:
            Gmax = 2 / N * (np.sqrt(N + 1) - 1)
        #
        rejected = True
        while rejected:
            x = np.random.rand()
            ratio = Gfunc(x, N) / Gmax
            test  = np.random.rand()
            if ratio > test:
                rejected = False
        #
        return np.arcsin(x)
        
    
    
def Gfunc(x, N):
    R2 = 2 * np.sqrt(N + 1)
    return x**(2 * N - 3) * np.abs(((N - 1) * R2 + 2) / N - R2 * x**2)
    
    
    
##############################
# Test
#
# TO BE WRITTEN

# import matplotlib.pyplot as plt

# sampler = SpinSampler()

# random_seed = 1234
# np.random.seed(random_seed)
# dist1 = []
# for i in range(100000):
#     dist1.append(sampler.get_condition(5))
    

# sampler = SpinSampler()

# random_seed = 1234
# dist2 = []
# for i in range(100000):
#     np.random.seed(random_seed + i)
#     dist2.append(sampler.get_condition(5))    

# plt.hist(dist1, bins = 50)
# plt.hist(dist2, bins = 50)