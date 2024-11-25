import numpy as np
from pysurf.spp import Model
from pysurf.sampling.normalmodes import Mode
from pysurf.spp.request import Request
import os.path
from scipy.interpolate import UnivariateSpline

class SumOfProductsPES(Model):
    """
    Analytic model potential energy surfaces
    """
    
    _user_input = """
    # File of the potential energy surface parameters
    parameter_file =  :: str
    """
    
    implemented = ['energy', 'gradient', 'nacs', 'time']  
    
    au2fs = 0.02418884
    
    @classmethod
    def from_config(cls, config):
        parameter_file = config['parameter_file']
        return cls.from_parameter_file(parameter_file)
    
    @classmethod
    def from_parameter_file(cls, parameter_file):
        with open(parameter_file, 'r') as pes_file:
            pes_data = pes_file.readlines()
        
        # Read the potential terms
        freq = []
        freq_ij = []  # accounts for kinematic couplings
        pes  = []
        func_def = []
        nstates = get_nstates(pes_data)  # get the number of electronic states
        for line in pes_data:
            add_potential_term(line, pes, func_def, nstates, freq, freq_ij)
            
        # Diagonalize the metric matrix and get the transformation matrix
        nModes = len(freq)
        if len(freq_ij) == 0:
            trafo = False
            C_trafo = np.eye(nModes)
        else:
            trafo = True
            omega = np.zeros((nModes,nModes))
            
            for f in freq:
                i = f[0]
                omega[i,i] = f[1]
            for f in freq_ij:
                i, j = f[0], f[1]
                omega[i,j] = 0.5 * f[2]
                omega[j,i] = 0.5 * f[2]
            
            w, C_trafo = np.linalg.eigh(omega)
            for i in range(nModes):
                freq[i] = (i, w[i])
            
        # Generate the functions to be evaluated and their derivatives from their definition
        path = os.path.abspath(parameter_file)
        path = os.path.dirname(path)
        func, d_func = generate_functions(func_def, path)
            
        #
        return cls(pes, nstates, freq, trafo, C_trafo, func, d_func)
    
    def __init__(self, pes, nstates, freq, trafo, C_trafo, func, d_func):
        self.pes = pes
        self.func = func
        self.d_func = d_func
        self.nstates = nstates
        self.nmodes = max(max(term[2]) for term in pes if len(term[2]) != 0) + 1
        self.freq = np.ones(self.nmodes)  # the default frequency is one
        for iM, w in freq:
            self.freq[iM] = w
        #
        self.trafo = trafo      # Should the input coordinates be transformed before evaluating the PES?
        self.C_trafo = C_trafo  # Transformation matrix
            
        # Attributes needed for sampling
        self.crd    = np.zeros(self.nmodes)
        disp  = np.eye(self.nmodes)
        self.modes  = [Mode(w, disp[i]) for i, w in enumerate(self.freq)]
        self.masses = 1.0 / self.freq.copy()
        
        # Time (needed for time-dependent potentials)
        self.time = 0.0
        
        #
        # Initialize the evaluation of the gradients
        #
        self.nonzero_gradient_terms = [[] for iM in range(self.nmodes)] 
        for iC in range(len(self.pes)):
            for iM in self.pes[iC][2]:
                if iM >= 0:
                    self.nonzero_gradient_terms[iM].append(iC)
            
    #################
    # PROPERTIES
    #################
    
    # Diabatic electronic Hamiltonian
    def _diab_Hel(self, Q):
        Hel = np.zeros((self.nstates, self.nstates))
        for term in self.pes:
            i, j = term[1]
            # each term involves a product of single coordinate functions
            dummy = 1.0
            time_factor = 1.0
            for k, l in zip(term[2], term[3]):
                if k == -1:
                    time_factor = self.func[l](self.time * self.au2fs)  # The parameters are given by input in fs
                else:
                    dummy *= self.func[l](Q[k])
                
            #
            Hel[i,j] += term[0] * dummy * time_factor
        #
        return Hel
    
    # Energy
    def _energy(self, Q, adiabatic = True):
        # Electronic Hamiltonian
        Hel = self._diab_Hel(Q)
        #
        return np.linalg.eigvalsh(Hel) if adiabatic else Hel.diagonal()
        
    # Gradient of the diabatic electronic Hamiltonian
    def _diab_Hel_gradient(self, Q):
        gradient = np.zeros((self.nstates, self.nstates, self.nmodes))
        #
        for iM in range(self.nmodes):
            for iC in self.nonzero_gradient_terms[iM]:
                term = self.pes[iC]
                # electronic states
                i, j = term[1]
                # get the position of the mode iM in the arrays of the active modes 
                idx = int(np.where(term[2] == iM)[0])
                # evaluate the derivative of the function for mode iM
                dummy = self.d_func[term[3][idx]](Q[iM])
                time_factor = 1.0
                for k, l in zip(np.concatenate((term[2][:idx], term[2][idx + 1:])),
                                np.concatenate((term[3][:idx], term[3][idx + 1:]))):
                    if k == -1:
                        time_factor = self.func[l](self.time * self.au2fs)
                    else:
                        dummy *= self.func[l](Q[k])
                #
                gradient[i,j,iM] += term[0] * dummy * time_factor
        #
        return gradient
    
    # Gradient
    def _gradient(self, Q, adiabatic = True):
        # Compute the gradient of the diabatic matrix
        gradient = self._diab_Hel_gradient(Q)
        #
        # Return the intra-state gradient
        # 
        if not adiabatic:
            return np.array([gradient[i,i] for i in range(self.nstates)])
        #
        else:
            # Compute the adiabatic energies
            Hel = self._diab_Hel(Q)
            V, C = np.linalg.eigh(Hel)
            # Adiabatic gradient
            gradient_ad = np.zeros((self.nstates, self.nmodes))
            for i in range(self.nstates):
                dummy = np.zeros(self.nmodes)
                for j in range(self.nstates):
                    for k in range(self.nstates):
                        dummy += C[j,i] * C[k,i] * gradient[j,k]
                #
                gradient_ad[i] = dummy.copy()
            #
            return gradient_ad
        
    def _nacs(self, Q):
        # Compute the gradient of the diabatic matrix
        gradient = self._diab_Hel_gradient(Q)
        # Compute the adiabatic energies
        Hel = self._diab_Hel(Q)
        V, C = np.linalg.eigh(Hel)
        # Nonadiabatic coupling
        nac = np.zeros((self.nstates, self.nstates, self.nmodes))
        for i in range(self.nstates):
            for j in range(self.nstates):
                dummy = np.zeros(self.nmodes)
                if i != j:
                    for k in range(self.nstates):
                        for l in range(self.nstates):
                            dummy += C[k,i] * C[l,j] * gradient[k,l]
                    dummy = - dummy / (V[i] - V[j])
                #
                nac[i,j] = dummy.copy()
        #
        return nac      

    # Request method: redirect to get
    def request(self, Q, properties, states = None, time = 0.0):
        if states == None:
            req = Request(Q, properties, states = list(range(self.nstates)))
        else:
            req = Request(Q, properties, states)   
        return self.get(req, time)
            
    # Get the requested properties
    def get(self, request, time = 0.0):
        """
        Parameters
        ----------
        request : dict
            Dictionary containing the values of the coordinates ('crd')
            and the requested properties.
            
            If diabatic properties are desired (instead of the default adiabatic ones)
            the field 'adiabatic' should be given and set to False

        Returns
        -------
        request : dict
            Updated dictionary with the calculated properties
        """
        #
        if self.trafo:
            Q = self.C_trafo @ request.crd 
        else:
            Q = request.crd
        #
        adiabatic = True
        if 'adiabatic' in request.keys():
            adiabatic = request['adiabatic']
        #
        for prop in request:
            if prop == 'energy':
                energy = self._energy(Q, adiabatic = adiabatic)
                request.set('energy', energy[request.states])
            if prop == 'gradient':
                grad = self._gradient(Q, adiabatic = adiabatic)
                if self.trafo:
                    grad = grad @ self.C_trafo
                request.set('gradient', grad[request.states,:])
            if prop == 'nacs':
                nacs_full = self._nacs(Q)
                if self.trafo:
                    nacs_full = nacs_full @ self.C_trafo
                nacs = {}
                for i in request.states:
                    for j in request.states:
                        if i != j:
                            nacs[(i,j)] = nacs_full[i,j]
                request.set('nacs', nacs)
            if prop == 'Hel':
                request.set('Hel', self._diab_Hel(Q))
            if prop == 'Hel_gradient':
                grad_full = self._diab_Hel_gradient(Q)
                if self.trafo:
                    grad_full = grad_full @ self.C_trafo
                #grad = {}
                #for i in request.states:
                #    for j in request.states:
                #        grad[(i,j)] = grad_full[i,j]
                request.set('Hel_gradient', grad_full)
                #request.set('Hel_gradient', grad)
        #
        #return None
        return request
    

    
##################################################
# Functions to read the potential input file

def get_nstates(pes_data):
    """
    Fetch the number of electronic states from the input potential file.
    
    Parameters
    ----------
    pes_data : list of str
        Lines read from the potential file.

    Returns
    -------
    nstates : int
        Number of electronic states.

    """
    nstates = 1
    for line in pes_data:
        # Remove comments and continue if its a blank line
        if '#' in line: line = line[:line.find('#')]
        if len(''.join(line.split())) == 0: continue
        #
        ls = line.replace('|',' ').split()
        if 'el' not in ls: continue   # electronic identity: no information
        i = ls.index('el')
        states = [int(s) for s in ls[i + 1][1:].split('&')]
        nstates = max(nstates, max(states))
    #
    return nstates


def add_potential_term(line, pes, func_def, nstates, freq, freq_ij):
    """    
    Parameters
    ----------
    line : str
        A line in the PES input file
        
    pes : list
        List of potential energy terms. .
        It is updated with the term read from line
        
    func_def : list
        Textual function defintion
        
    nstates : int
        The number of electronic states.
    
    Returns
    -------
    None.
    """
    # Remove comments and return if its a blank line
    if '#' in line: line = line[:line.find('#')]
    if len(line.strip()) == 0: return None
    # Split into product terms
    ls = line.split('|')
    
    #
    # FIRST CASE: The line contains info about masses/frequencies
    #     
    
    if ls[0].strip() in ('freq', 'mass'):
        for term in ls[1:]:
            ts = term.split()
            # check that the format is right
            if len(ts) not in (2,3):
                print('ERROR: the frequency/mass terms should be in the form:')
                print('   | mode value')
                print('or')
                print('   | mode1 mode2 value')
                exit()
                
            # Read the modes and the associated mass/frequency
            
            if len(ts) == 2:
                dummy = float(ts[1]) if ls[0].strip() == 'freq' else 1.0 / float(ts[1])
                freq.append( (int(ts[0]) - 1, dummy) )
                
            if len(ts) == 3:   # kinematic couplings
                dummy = float(ts[2]) if ls[0].strip() == 'freq' else 1.0 / float(ts[2])
                i1 = int(ts[0]) - 1
                i2 = int(ts[1]) - 1
                freq_ij.append( (i1,i2,dummy) )
        #            
        return None
    
    #
    # SECOND CASE: The line contains a term of the potential
    #
    
    # The first term is the coefficient
    coeff = float(ls[0])
    # other factors
    iMode = []
    iFunc = []
    electronic_identity = True
    for term in ls[1:]:
        ts = term.split()
        # check that the format is right
        if len(ts) < 2:
            print('ERROR: the product terms should be in the form "| mode function"')
            print()
            print(line)
            exit()
            
        #
        # Read the electronic operator and/or the potential factors
        #
        
        if ts[0] == 'el':  # electronic operator
        
            if ts[1] != '1':
                electronic_identity = False
                if 'S' not in ts[1] or '&' not in ts[1]:
                    print('ERROR: the electronic operator should be in the form Si&j')
                    exit()
                #
                states = [int(i) - 1 for i in ts[1][1:].split('&')]
                
        else:  # coordinate function
        
            # Skip to the next term if identity
            if ts[1] == '1':
                continue
            # Mode
            mode_idx = int(ts[0])
            iMode.append(mode_idx - 1)
            # Function
            func = ts[1:]
            try:
                idx = func_def.index(func)
            except ValueError:
                func_def.append(func)
                idx = len(func_def) - 1
            iFunc.append(idx)
    #
    
    iMode = np.array(iMode, dtype = np.int32)
    iFunc = np.array(iFunc, dtype = np.int32)

    #
    if electronic_identity:
        for i in range(nstates):
            pes.append([coeff, (i,i), iMode, iFunc])
    else:
        pes.append([coeff, tuple(states), iMode, iFunc])
        if states[0] != states[1]:
            pes.append([coeff, (states[1], states[0]), iMode, iFunc])
    

##################################################
# Generate analytic functions from their definition

def generate_functions(func_def, path):
    func   = []
    d_func = []
    
    iC = 0
    for f in func_def:
        # f has the form
        # [function name, parameter 1, parameter 2, ...]
        
        if 'q^' in f[0]:
            n = int(f[0][2:])
            func.append(lambda x, n = n: x**n)     #  The value of the parameters needs to be "captured", otherwise it might get taken from the global variables
            d_func.append(lambda x, n = n: n * x**(n - 1))
            iC += 1
            continue
        
        if 'q' in f[0]:
            func.append(lambda x: x)
            d_func.append(lambda x: 1)
        
        if f[0] == 'sin':   # sin(a * x + b)^n
            if len(f) != 4:
                print('The function sin(a * x + b)^n should be given as')
                print('  sin a b n')
                print('Some parameter is missing')
            #
            a = float(f[1])
            b = float(f[2])
            n = int(f[3])
            #
            func.append(lambda x, a = a, b = b, n = n:
                        np.sin(a * x + b)**n)
            d_func.append(lambda x, a = a, b = b, n = n:
                          a * n * np.sin(a * x + b)**(n - 1) * np.cos(a * x + b))
            
        if f[0] == 'cos':   # sin(a * x + b)^n
            if len(f) != 4:
                print('The function cos(a * x + b)^n should be given as')
                print('  cos a b n')
                print('Some parameter is missing')
            #
            a = float(f[1])
            b = float(f[2])
            n = int(f[3])
            #
            func.append(lambda x, a = a, b = b, n = n:
                        np.cos(a * x + b)**n)
            d_func.append(lambda x, a = a, b = b, n = n:
                          - a * n * np.cos(a * x + b)**(n - 1) * np.sin(a * x + b))
            
        if f[0].lower() == 'morse':
            if len(f) < 5:
                print('The function V + D * (1 - exp(-k * (r - r0)))^2 should be given as')
                print('   Morse V D k r0')
                print('Some parameter is missing')
            #
            V  = float(f[1])
            D  = float(f[2])
            k  = float(f[3])
            r0 = float(f[4])
            #
            func.append(lambda x, V = V, D = D, k = k, r0 = r0:
                        V + D * (1 - np.exp(-k * (x - r0)))**2)
            d_func.append(lambda x, V = V, D = D, k = k, r0 = r0:
                          2 * D * (1 - np.exp(-k * (x - r0))) * k * np.exp(-k * (x - r0)))
                
        if f[0].lower() == 'gau':
            if len(f) < 3:
                print('The function exp(-0.5 * (x - x0)^2 / delta^2) should be given as')
                print('   Gau x0 delta')
                print('Some parameter is missing')
            #
            x0    = float(f[1])
            delta = float(f[2])
            #
            func.append(lambda x, x0 = x0, delta = delta:
                        np.exp(-0.5 * ((x - x0) / delta)**2))
            d_func.append(lambda x, x0 = x0, delta = delta:
                          - (x - x0) / delta**2 * np.exp(-0.5 * ((x - x0) / delta)**2))
            
        if f[0].lower() == 'interp':
            # Function interpolated on a grid
            data_file = os.path.join(path, f[1])
            with open(data_file, 'r') as fh:
                interp_data = fh.readlines()
            #
            nP = len(interp_data)
            x, y = np.zeros(nP), np.zeros(nP)
            for i in range(nP):
                x[i] = float(interp_data[i].split()[0])
                y[i] = float(interp_data[i].split()[1])
            #
            spl = UnivariateSpline(x, y, k = 2, s = 0)
            func.append(spl)
            d_func.append(spl.derivative(1))
                
        ###  
        iC += 1
            
    ###
    return func, d_func

    
##############################
if __name__ == '__main__':
    sop = SumOfProductsPES.from_parameter_file('pyrmod6.pes')
    print(sop.nstates)
    print(sop.nmodes)
    # TEST 1: Request energy and gradient
    Q = [0.1, -0.5, 2.1, 0.5, 0.6, -1.0]
    from pysurf.spp.request import Request
    req = Request(Q, ['energy', 'gradient', 'nacs'], states = range(sop.nstates))   
    sop.get(req)   
    print(r['energy'])
    # TEST 2: Sampling
    from pysurf.sampling import Wigner
    sampling = Wigner.from_model(sop)
    print(sampling.get_condition())
           
        
        
    
    
    
    