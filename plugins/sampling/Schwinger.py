import numpy as np
from pysurf.sampling.base_sampler import DynSamplerBase

class Schwinger(DynSamplerBase):
    """
    One-dimensional Schwinger sampler using inverse transform sampling
    """
    @classmethod
    def from_config(cls, config, start=None):
        pass
    
    def get_condition(self):
        r = get_r_condition()
        phi = np.random.random() * 2 * np.pi
        cond = DynSamplerBase.condition([r * np.cos(phi)], [r * np.sin(phi)], 0)
        return cond
    
    
def get_r_condition():
    """
    Sample a value of r from the distribution
       F(r) = N * |r^2 - 1/2| * r * exp(-r^2)
    by inversion sampling from the distribution for the variable z = r^2
       G(z) = C * |z - 1/2| * exp(-z)
    """    
    eps = 1e-8
    #
    # Random number in the range [0,1]
    #
    y = np.random.random()
    #
    # Use the bisection method to find a value of the 
    # cumulative distribution function CDF such that CDF = y
    #
    ### Define the initial interval
    z0 = 0.0
    z1 = 10.0
    while G_CDF(z1) < y:
        z1 += 1.0
    ### Bisect
    while z1 - z0 > eps:
        zave = 0.5 * (z0 + z1)
        if G_CDF(zave) < y:
            z0 = zave
        else:
            z1 = zave
    #
    return np.sqrt(zave)

def G_CDF(z):
    """
    Cumulative distribution function for the distribution
       G(z) = C * |z - 1/2| * exp(-z)
    """
    sqrt_e = np.sqrt(np.e)
    C = 2 * sqrt_e / (4 - sqrt_e)
    dummy = np.exp(-z) * (1 + 2 * z)
    return 0.5 * C * (dummy - 1) if z < 0.5 else 1 - 0.5 * C * dummy


##############################
# Test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    S_samp = Schwinger()
    
    r = []
    x = []
    p = []
    for i in range(20000):
        cond = S_samp.get_condition()
        x.append(cond.crd[0])
        p.append(cond.veloc[0])
        r.append(np.sqrt(cond.crd[0]**2 + cond.veloc[0]**2))
    #
    plt.hist(r, bins = 200)
    plt.show()
    plt.hist(x, bins = 200)
    plt.show()
    plt.hist(p, bins = 200)
    plt.show()




    