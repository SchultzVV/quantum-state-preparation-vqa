from src.simulation_with_save import Simulate
from kraus_maps import QuantumChannels as QCH
import numpy as np
from numpy import pi


n_qubits = 2
d_rho_A = 2
list_p = np.linspace(0.001,60,3)
epochs = 1
step_to_start = 1
rho_AB = QCH.rho_AB_ad
S = Simulate('ad', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
#list_p = S.get_list_p_noMarkov()
# print(list_p)
# print(type(list_p))
S.run_calcs_noMarkov(True, pi/2, 0)