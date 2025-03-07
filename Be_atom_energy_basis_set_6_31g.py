#% This python code computes the energies for ground and first excited states for beryllium atom using quantum computing [1, 2]
#% and the 6-31G basis set. Atomic units are used in the calculation, and the ground state energy is compared with that from 
#% the full configuration interaction (FCI) [3].
#%
#% Refs: [1] https://github.com/quantumlib/OpenFermion-PySCF
#%       [2] https://quantumai.google/openfermion/tutorials/intro_to_openfermion
#%       [3] https://cccbdb.nist.gov/energy2x.asp
#%
#% Refs: [1] https://github.com/quantumlib/OpenFermion-PySCF 
#%       [2] https://quantumai.google/openfermion/tutorials/intro_to_openfermion
#%
#% Written by Tsogbayar Tsednee (PhD)
#%
#% Email: tsog215@gmail.com
#% March 3, 2024 & University of North Dakota 

###
# import libraries
#
import numpy as np
from scipy.sparse import linalg
#
import cirq
import openfermion as of
import openfermionpyscf as ofpyscf
###


# Set atom's parameters
geometry = [('Be', (0.0, 0.0, 0.0))]
basis = '6-31g'
multiplicity = 1
charge = 0

# Perform electronic structure calculations and
# obtain Hamiltonian as an InteractionOperator
hamiltonian = ofpyscf.generate_molecular_hamiltonian(
            geometry, basis, multiplicity, charge)
#
#print(hamiltonian)

# Convert to a FermionOperator
hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)

# Map to QubitOperator using the Bravy-Kitaev transformation
hamiltonian_bk = of.bravyi_kitaev(hamiltonian_ferm_op)

# Convert to Scipy sparse matrix
hamiltonian_bk_sparse = of.get_sparse_operator(hamiltonian_bk)

# Compute ground energy
eigs, _ = linalg.eigsh(hamiltonian_bk_sparse, k=1, which='SA')
ground_state_energy = eigs[0]

# Compute the first excited energy
eigs, _ = linalg.eigsh(hamiltonian_bk_sparse, k=2, which='SA')
first_excited_state_energy = eigs[1]


###
E0 = print(ground_state_energy)        # E0 = -14.613545269594411 vs -14.613544 = FCI [3]
#
E1 = print(first_excited_state_energy) # E1 = -14.528756088041364   
###


