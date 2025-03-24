import numpy as np
#from ..spp import SurfacePointProvider
from pysurf.dynamics.base_propagator import PropagatorBase
#from pysurf.spp.request import Request
from pysurf.utils.constants import fs2au

#############################################

class QuasiClassicalMapping(PropagatorBase):
    
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
        
        n_output = round(self.tout / self.time_step)
        
        #
        # For the adiabatic representation, the size of the mass array
        # should be consisten with the size of Pdot
        #
        if self.state_repr == 'adiabatic':
            if len(self.vel.shape) == 1:
                mass_P = self.masses
            else:
                mass_P = np.array([self.masses] * self.vel.shape[-1]).T
        #
        
        for istep in range(self.start, nsteps + 1):
        
            time = self.dt * istep
            if 'time' in self.spp._interface.implemented:
                self.spp.time = time
            
            #
            # DIABATIC REPRESENTATION
            #
            if self.state_repr == 'diabatic':
                # Get the electronic Hamiltonian and its gradient
                data = self.call_spp(crd = self.crd)
                Hel        = data['Hel']
                Hel_grad_i = data['Hel_gradient']
                
                # (1) Update electronic coordinates for the first half-step
                x, p = elec_step(self.x, self.p, Hel, self.dt * 0.5)
                self.x = x
                self.p = p
                
                # (2) Update the nuclear coordinates
                Pdot = coord_step(self.x, self.p, Hel_grad_i, self.dt)
                self.crd += self.dt * self.vel + self.dt**2 * 0.5 * Pdot / self.masses
                
                # (3) Update the velocities
                data = self.call_spp(crd = self.crd)
                Hel        = data['Hel']   # needed for the last step
                Hel_grad_f = data['Hel_gradient']
                
                Pdot = veloc_step(self.x, self.p, Hel_grad_i, Hel_grad_f, self.dt)
                self.vel += self.dt * Pdot / self.masses
                
                # (4) Update the electronic coordinates for the second half-step
                x, p = elec_step(self.x, self.p, Hel, self.dt * 0.5)
                self.x = x
                self.p = p
                
            #
            # ADIABATIC REPRESENTATION
            # 
            if self.state_repr == 'adiabatic':
                # Get the electronic Hamiltonian, its gradient and the nacs
                data = self.call_spp(crd = self.crd)
                Hel_i = data['energy']
                Hel_grad_i = data['gradient'].data
                nacs_i = np.zeros((self.nstates, self.nstates) + self.crd.shape)
                for n, m in data['nacs']:
                    nacs_i[n,m] = data['nacs'][(n,m)]
                    
                # (1) Update electronic coordinates for the first half-step
                x, p = elec_step_adiab(self.x, self.p, Hel_i, nacs_i, self.vel, self.dt * 0.5)
                self.x = x
                self.p = p
                
                # (2) Update the nuclear coordinates
                Pdot = coord_step_adiab(self.x, self.p, Hel_i, Hel_grad_i, nacs_i, self.dt)
                self.crd += self.dt * self.vel + self.dt**2 * 0.5 * Pdot / mass_P
                
                # (3) Update the velocities
                data = self.call_spp(crd = self.crd)
                Hel_f = data['energy']
                Hel_grad_f = data['gradient'].data
                nacs_f = np.zeros((self.nstates, self.nstates) + self.crd.shape)
                for n, m in data['nacs']:
                    nacs_f[n,m] = data['nacs'][(n,m)]
                    
                Pdot = veloc_step_adiab(self.x, self.p, Hel_i, Hel_f, Hel_grad_i, Hel_grad_f, nacs_i, nacs_f, self.dt)
                self.vel += self.dt * Pdot / mass_P
                
                # (4) Update the electronic coordinates for the second half-step
                x, p = elec_step_adiab(self.x, self.p, Hel_f, nacs_f, self.vel, self.dt * 0.5)
                self.x = x
                self.p = p
                
                Hel = np.diag(Hel_f)
            
            #####################
            # Calculate the observables for the standard output
            self.ekin = calc_eKin(self.masses, self.vel)
            self.elec = calc_eElec(self.x, self.p, Hel)
            self.etot = self.ekin + self.elec
            
            # Write step info
            self.output_step(istep, time, self.ekin, self.elec, self.etot)   ### Check
            
            # Add the data to the database
            if np.mod(istep, n_output) == 0:
                self.db.append('time', (time / fs2au))
                self.db.append('crd' , self.crd)
                self.db.append('vel' , self.vel)
                self.db.append('elec', np.c_[self.x, self.p])
                self.db.append('Hel' , Hel)
                self.db.increase

    def call_spp(self, crd):
        data = self.spp.request(crd, self.properties)
        #req = Request(crd, self.properties, list(range(self.nstates)))
        #self.spp.get(req)
        #
        return data

    def setup_new(self):
        # set starting coordinates and velocities
        self.crd = self.init.crd
        self.vel = self.init.veloc
        
        # Get the electronic Hamiltonian and its gradient
        data = self.call_spp(crd = self.crd)
        Hel = data['Hel'] if self.state_repr == 'diabatic' else np.diag(data['energy'])
        
        # Calculate the observables for the standard output
        self.ekin = calc_eKin(self.masses, self.vel)
        self.elec = calc_eElec(self.x, self.p, Hel)        
        self.etot = self.ekin + self.elec
        
        popA, popB = calc_pop(self.x, self.p)
        
        # Put initial condition as step 0 into database
        istep = 0
        time  = self.dt * istep
        self.output_step(istep, time, self.ekin, self.elec, self.etot) 
        self.db.append('time', time)
        self.db.append('crd' , self.crd)
        self.db.append('vel' , self.vel)
        self.db.append('elec', np.c_[self.x, self.p])
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
            self.vel = self.db.get('veloc', -1)



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



def elec_step_adiab(x, p, Hel, nacs, vel, dt):
    """
    Propagation step for the electronic coordinates and momenta

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta
    Hel : N-dimensional array
        Electronic Hamiltonian matrx
    nacs : N x N x n_modes
        Nonadiabatic couplings
    vel : nmodes
        Velocities
    dt : float
        Step size

    Returns
    -------
    x, p : (array,array)
        Updated coordinate and velocity vectors

    """
    Ns = len(x)
    
    H_ave = np.sum(Hel) / Ns
    V = np.diag(Hel - H_ave)
    
    F = np.zeros((Ns,Ns))
    for n in range(Ns):
        for m in range(Ns):
            F[n,m] = np.sum(vel * nacs[n,m])
            
    #
    A = V - 1j * F
    w, U = np.linalg.eigh(A)
    cosA = U @ np.diag(np.cos(w * dt)) @ np.conj(U.T)
    sinA = U @ np.diag(np.sin(w * dt)) @ np.conj(U.T)
    
    #
    x_new = (cosA @ x + sinA @ p).real + (sinA @ x - cosA @ p).imag
    p_new = (cosA @ x + sinA @ p).imag - (sinA @ x - cosA @ p).real
    
    #
    return x_new, p_new



def coord_step_adiab(x, p, V, Hel_grad, nacs, dt):
    """
    Propagation step for the nuclear coordinates

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta
    V : N-dimensional array
        Adiabatic energies 
    Hel_grad : N x ncoo array
        Gradients of the electronic Hamiltonian
    nacs : N x N x ncoo array
        Derivative couplings
    dt : float
        Step size

    Returns
    -------
    crd : array
        Updated coordinate vectors

    """
    Ns = len(x)
    ncoo = Hel_grad.shape[1:]
    
    ElecZPE = (np.dot(x,x) + np.dot(p,p)) * 0.5
    shift = (1 - ElecZPE) / Ns
    
    # Born-Oppenheimer contribution
    Pdot = np.zeros(ncoo)    
    for n in range(Ns):
        Pdot -= Hel_grad[n] * (0.5 * (x[n]**2 + p[n]**2) + shift)
        
    # Derivative coupling contribution
    for n in range(Ns):
        for m in range(Ns):
            Pdot += 0.5 * (x[n] * x[m] + p[n] * p[m]) * nacs[n,m] * (V[n] - V[m])
        
    #
    return Pdot



def veloc_step_adiab(x, p, V_i, V_f, Hel_grad_i, Hel_grad_f, nacs_i, nacs_f, dt):
    """
    Propagation step for the nuclear coordinates

    Parameters
    ----------
    x : array
        Electronic coordinates
    p : array
        Electronic momenta
    V_i, V_f : N-dimensional arrays
        Adiabatic energies at the initial and final geometry
    Hel_grad_i, Hel_grad_f : N x N x ncoo array
        Gradients of the electronic Hamiltonian at the initial and final geometry
    dt : float
        Step size

    Returns
    -------
    crd : array
        Updated coordinate vectors

    """
    Ns = len(x)
    ncoo = Hel_grad_i.shape[1:]
    
    ElecZPE = (np.dot(x,x) + np.dot(p,p)) * 0.5
    shift = (1 - ElecZPE) / Ns
    
    # Average gradients
    Hel_grad = 0.5 * (Hel_grad_i + Hel_grad_f)
    
    # Average nac force
    nac_force = np.zeros_like(nacs_i)
    for n in range(Ns):
        for m in range(Ns):
            nac_force[n,m] = 0.5 * (nacs_i[n,m] * (V_i[n] - V_i[m]) \
                                 +  nacs_f[n,m] * (V_f[n] - V_f[m]) )
                
    # Born-Oppenheimer contribution
    Pdot = np.zeros(ncoo)    
    for n in range(Ns):
        Pdot -= Hel_grad[n] * (0.5 * (x[n]**2 + p[n]**2) + shift)
        
    # Derivative coupling contribution
    for n in range(Ns):
        for m in range(Ns):
            Pdot += 0.5 * (x[n] * x[m] + p[n] * p[m]) * nac_force[n,m]
    
    #
    return Pdot

