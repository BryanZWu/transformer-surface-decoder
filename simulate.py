# A simulator for a planar code with a single qubit using qiskit

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import time


# TODO: instead of defining this noise model, look into using the 
# noise model from real devices, like the ibmq_16_melbourne
# https://qiskit.org/documentation/stable/0.34/apidoc/aer_noise.html

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

def noisy_planar_code(n_physical_qubits, noise_model, n_cycles):
    '''
    Returns a noisy planar code circuit with a single logical qubit.
    We will be measuring syndrome for each cycle and collecting the results
    as data for the decoder.

    Args:
        n_physical_qubits: the number of physical qubits to use
        noise_model: the noise model to use
    '''
    pass # TODO: implement this

def seventeen_qubit_planar_code(num_cycles=1):
    '''
    Returns a seventeen qubit planar code circuit with a single logical qubit.

    See https://www.researchgate.net/figure/a-Planar-layout-of-the-17-qubit-surface-code-White-black-circles-represent-data_fig1_320223633
    And https://arxiv.org/pdf/1404.3747.pdf

    In the 17 qubit planar code, logical X can be X2 X4 X6, and logical Z can be Z0 Z4 Z8.
    According to the second link

    The definitions of syndrome bits are as follows:

    First, the X syndromes
    S0 = X1 X2
    S1 = X0 X1 X3 X4
    S2 = X4 X5 X7 X8
    S3 = X6 X7

    Then, the Z syndromes
    S4 = Z0 Z3
    S5 = Z1 Z2 Z4 Z5
    S6 = Z3 Z4 Z6 Z7
    S7 = Z5 Z8
    '''
    x_syndrome = {
        0: [1, 2],
        1: [0, 1, 3, 4],
        2: [4, 5, 7, 8],
        3: [6, 7]
    }
    z_syndrome = {
        4: [0, 3],
        5: [1, 2, 4, 5],
        6: [3, 4, 6, 7],
        7: [5, 8]
    }
    code_register = QuantumRegister(9, 'code_register')
    ancilla_register = QuantumRegister(8, 'ancilla_register')
    syndrome_register = ClassicalRegister(8 * num_cycles, 'syndrome_register')
    
    # define the circuit
    circuit = QuantumCircuit(code_register, ancilla_register, syndrome_register, name='seventeen_qubit_planar_code')

    # define ancilla qubits by adding gates. First, the X ancillas
    for ancilla in x_syndrome:
        for qubit in x_syndrome[ancilla]:
            circuit.cx(code_register[qubit], ancilla_register[ancilla])

    # Then, the Z ancillas
    for ancilla in z_syndrome:
        for qubit in z_syndrome[ancilla]:
            circuit.cz(code_register[qubit], ancilla_register[ancilla])
    
    n_syndromes = len(x_syndrome) + len(z_syndrome)
    for i in range(num_cycles):
        # Measure the X stabilizers
        for ancilla in x_syndrome:
            circuit.measure(ancilla_register[ancilla], syndrome_register[i * n_syndromes + ancilla])
        
        # Measure the Z stabilizers
        for ancilla in z_syndrome:
            circuit.measure(ancilla_register[ancilla], syndrome_register[i * n_syndromes + ancilla])

    return circuit

