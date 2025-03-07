#% This python code computes the energy for ground state for neon atom using quantum computing [1, 2]
#% and the 6-31G basis set. Atomic units are used in the calculation, and the ground state energy is compared with that from 
#% the full configuration interaction (FCI) [3].
#%
#% Refs: [1] https://github.com/quantumlib/OpenFermion-PySCF
#%       [2] https://quantumai.google/openfermion/tutorials/intro_to_openfermion
#%       [3] https://cccbdb.nist.gov/energy2x.asp
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
geometry = [('Ne', (0.0, 0.0, 0.0))]
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

# Map to QubitOperator using the JWT
hamiltonian_bk = of.bravyi_kitaev(hamiltonian_ferm_op)

# Convert to Scipy sparse matrix
hamiltonian_bk_sparse = of.get_sparse_operator(hamiltonian_bk)

# Compute ground energy
eigs, _ = linalg.eigsh(hamiltonian_bk_sparse, k=1, which='SA')
ground_state_energy = eigs[0]


###
E0 = print(ground_state_energy)        # E0 = -128.58980234988329 vs -128.589653 = FCI [3]
#
###

