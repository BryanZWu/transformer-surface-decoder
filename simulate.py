# A simulator for a planar code with a single qubit using qiskit

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np


def get_noise(p_meas,p_gate):
    '''
    Returns a noise model with the given parameters'''

    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

def seventeen_qubit_planar_code(n_cycles, noise_model):
    '''
    Returns a seventeen qubit planar code circuit with a single logical qubit.

    See https://www.researchgate.net/figure/a-Planar-layout-of-the-17-qubit-surface-code-White-black-circles-represent-data_fig1_320223633
    And https://arxiv.org/pdf/1404.3747.pdf

    In the 17 qubit planar code, logical X can be X2 X4 X6, and logical Z can be Z0 Z4 Z8.
    According to the second link
    '''
    code_register = QuantumRegister(9, 'code_register')
    ancilla_register = QuantumRegister(8, 'ancilla_register')
    syndrome_register = ClassicalRegister(8, 'syndrome_register')




def noisy_planar_code(n_physical_qubits, noise_model, n_cycles):
    '''
    Returns a noisy planar code circuit with a single logical qubit.
    We will be measuring syndrome for each cycle and collecting the results
    as data for the decoder.

    Args:
        n_physical_qubits: the number of physical qubits to use
        noise_model: the noise model to use
    '''
    n_rows = int(n_physical_qubits ** 0.5)
    if n_rows ** 2 != n_physical_qubits:
        raise ValueError('n_physical_qubits must be a square number')
    code_register = QuantumRegister(n_physical_qubits, 'code_register')

    # In a planar code, there is an ancilla qubit for each data qubit

    # 