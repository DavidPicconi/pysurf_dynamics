import numpy as np
#from ..spp import SurfacePointProvider
from pysurf.dynamics.base_propagator import PropagatorBase
from pysurf.spp.request import Request
from pysurf.utils.constants import fs2au

#############################################

class QuasiClassicalMapping(PropagatorBase):
    
    properties = ['Hel', 'Hel_gradient'] 
    
    def _run(self, nsteps, dt):
        self.dt = dt
        self.nsteps = nsteps
        
        if self.restart is False:
            self.setup_new()
            if self.nsteps < 1:
                return
        else:
            self.setup_from_db()

        if self.start > self.nsteps:
            self.logger.info("Dynamics already performed for more steps!")
            return 
        
        Ns = self.nstates
        n_output = round(self.tout / self.time_step)
        for istep in range(self.start, nsteps + 1):
            time = self.dt * istep
            if 'time' in self.spp.implemented:
                self.spp.time = time
            
            # Get the electronic Hamiltonian and its gradient
            data = self.call_spp(crd = self.crd[Ns:])
            Hel        = data['Hel']
            Hel_grad_i = data['Hel_gradient']
            
            # (1) Update electronic coordinates for the first half-step
            x, p = elec_step(self.crd[:Ns], self.v[:Ns],
                             Hel, self.dt * 0.5)
            self.crd[:Ns] = x
            self.v[:Ns] = p
            
            # (2) Update the nuclear coordinates
            Pdot = coord_step(self.crd[:Ns], self.v[:Ns], 
                              Hel_grad_i, self.dt)
            self.crd[Ns:] += self.dt * self.v[Ns:] \
                           + self.dt**2 * 0.5 * Pdot / self.masses
            
            # (3) Update the velocities
            data = self.call_spp(crd = self.crd[Ns:])
            Hel        = data['Hel']   # needed for the last step
            Hel_grad_f = data['Hel_gradient']
            Pdot = veloc_step(self.crd[:Ns], self.v[:Ns],
                              Hel_grad_i, Hel_grad_f, self.dt)
            self.v[Ns:] += self.dt * Pdot / self.masses
            
            # (4) Update the electronic coordinates for the second half-step
            x, p = elec_step(self.crd[:Ns], self.v[:Ns],
                             Hel, self.dt * 0.5)
            self.crd[:Ns] = x
            self.v[:Ns] = p
            
            #####################
            # Calculate the observables for the standard output
            self.ekin = calc_eKin(self.masses, self.v[Ns:])
            self.elec = calc_eElec(self.crd[:Ns],
                                   self.v[:Ns],
                                   Hel)
            self.etot = self.ekin + self.elec
            
            # Write step info
            self.output_step(istep, time, self.ekin, self.elec, self.etot)   ### Check
            
            # Add the data to the database
            #self.db.add_step(time, self.crd, self.v, Hel, popA, popB)
            if np.mod(istep, n_output) == 0:
                self.db.append('time', (time / fs2au))
                self.db.append('crd' , self.crd[Ns:])
                self.db.append('vel' ,   self.v[Ns:])
                self.db.append('elec', np.c_[self.crd[:Ns], self.v[:Ns]])
                self.db.append('Hel' , Hel)
                self.db.increase

    def call_spp(self, crd):
        req = Request(crd, self.properties, list(range(self.nstates)))
        self.spp.get(req)
        #
        return req

    def setup_new(self):
        # set starting coordinates and velocities
        self.crd = self.init.crd
        self.v = self.init.veloc
        
        # Get the electronic Hamiltonian and its gradient
        data = self.call_spp(crd = self.crd[self.nstates:])
        Hel  = data['Hel']
        
        # Calculate the observables for the standard output
        self.ekin = calc_eKin(self.masses, self.v[self.nstates:])
        self.elec = calc_eElec(self.crd[:self.nstates],
                               self.v[:self.nstates],
                               Hel)
        self.etot = self.ekin + self.elec
        popA, popB = calc_pop(self.crd[:self.nstates], self.v[:self.nstates])
        
        # Put initial condition as step 0 into database
        istep = 0
        time  = self.dt * istep
        self.output_step(istep, time, self.ekin, self.elec, self.etot) 
        self.db.append('time', time)
        self.db.append('crd' , self.crd[self.nstates:])
        self.db.append('vel' ,   self.v[self.nstates:])
        self.db.append('elec', np.c_[self.crd[:self.nstates], self.v[:self.nstates]])
        self.db.append('Hel' , Hel)
        self.db.increase
        #
        self.start = 1
        
    def setup_from_db(self):
        # at least one previous step is needed
        if len(self.db) < 1:
            self.create_new_db()
            return self.setup_new()
        else:
            self.crd = self.db.get('crd', -1)
            self.v = self.db.get('veloc', -1)



def calc_eKin(masses, veloc):
    """
    Calculate the nuclear kinetic energy
    """
    ekin = 0.0
    #Take care that for an ab-initio calculation masses are a 1D array of length natoms
    #and velocities are a 2D array of shape (natoms, 3)
    if veloc.shape != masses.shape:
        masses_new = np.repeat(masses, 3).reshape((len(masses),3))
    else:
        masses_new = masses
    for i, mass in enumerate(masses_new.flatten()):
        ekin += 0.5 * mass * veloc.flatten()[i] ** 2
    return ekin


def calc_eElec(x, p, Hel):
    """
    Calculate the electronic energy

    Parameters
    ----------
    x   : array
        Coordinates of the electronic degrees of freedom
    p   : array
        Momenta of the electronic degrees of freedom
    Hel : N x N matrix
          electronic Hamiltonian

    Returns
    -------
    eElec : float
            Electronic energy
            
    """
    ElecZPE = (np.dot(x,x) + np.dot(p,p)) * 0.5
    return (1 - ElecZPE) / len(x) * np.trace(Hel) \
        + 0.5 * (np.dot(p, Hel @ p) + np.dot(x, Hel @ x))
        

def calc_pop(x, p):
    """
    Calculate the value of the electronic population observables

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta

    Returns
    -------
    pop : array containing the population for each state

    """
    Ns = len(x)
    # Gaussian factor
    Gau = np.exp(- np.dot(x,x) - np.dot(p,p)) * 2**(Ns + 1)
    x2p2 = x**2 + p**2
    #
    return (x2p2 - 0.5) * Gau, 0.5 * (x2p2 - 1.0)
    #return 0.5 * (x**2 + p**2 - 1.0)
    #return (x**2 + p**2 - 0.5) * Gau
    
    

def elec_step(x, p, Hel, dt):
    """
    Propagation step for the electronic coordinates and momenta

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta
    Hel : N x N array
        Electronic Hamiltonian matrx
    dt : float
        Step size

    Returns
    -------
    x, p : (array,array)
        Updated coordinate and velocity vectors

    """
    Ns = len(x)
    #
    H_ave = np.trace(Hel) / Ns
    V, C = np.linalg.eigh(Hel - np.eye(Ns) * H_ave)
    cosHel = C @ np.diag(np.cos(V * dt)) @ C.T
    sinHel = C @ np.diag(np.sin(V * dt)) @ C.T
    #
    x_new =   cosHel @ x + sinHel @ p
    p_new = - sinHel @ x + cosHel @ p
    #
    return x_new, p_new



def coord_step(x, p, Hel_grad, dt):
    """
    Propagation step for the nuclear coordinates

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta
    Hel_grad : N x N x ncoo array
        Gradients of the electronic Hamiltonian
    dt : float
        Step size

    Returns
    -------
    crd : array
        Updated coordinate vectors

    """
    Ns = len(x)
    ncoo = Hel_grad.shape[-1]
    #
    ElecZPE = (np.dot(x,x) + np.dot(p,p)) * 0.5
    Pdot = np.zeros(ncoo)
    for k in range(ncoo):
        Pdot[k] = (ElecZPE - 1) / Ns * np.trace(Hel_grad[:,:,k]) \
                - 0.5 * (np.dot(p, Hel_grad[:,:,k] @ p) + np.dot(x, Hel_grad[:,:,k] @ x))
    #
    return Pdot



def veloc_step(x, p, Hel_grad_i, Hel_grad_f, dt):
    """
    Propagation step for the nuclear coordinates

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta
    Hel_grad_i, Hel_grad_f : N x N x ncoo array
        Gradients of the electronic Hamiltonian at the initial and final geometry
    dt : float
        Step size

    Returns
    -------
    crd : array
        Updated coordinate vectors

    """
    # Average gradients
    Hel_grad = 0.5 * (Hel_grad_i + Hel_grad_f)
    #
    Ns = len(x)
    ncoo = Hel_grad.shape[-1]
    #
    ElecZPE = (np.dot(x,x) + np.dot(p,p)) * 0.5
    Pdot = np.zeros(ncoo)
    for k in range(ncoo):
        Pdot[k] = (ElecZPE - 1) / Ns * np.trace(Hel_grad[:,:,k]) \
                - 0.5 * (np.dot(p, Hel_grad[:,:,k] @ p) + np.dot(x, Hel_grad[:,:,k] @ x))
    #
    return Pdot

