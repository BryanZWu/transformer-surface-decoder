import os
import pandas as pd
from torch.utils.data import Dataset
import random
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from simulate import seventeen_qubit_planar_code, get_noise
import torch
import time

class SurfaceCodeDataset(Dataset):
    """
    A dataset for quantum error correction using surface codes. 
    Simulation is done using qiskit, data generated on demand (not stored on disk).
    """
    def __init__(self, noise_model):
        self.noise_model = noise_model

    def __len__(self):
        return -1 # Infinite dataset, just simulate on demand
        # return 100 # Random fixed size dataset

    def __getitem__(self, idx):
        '''Just simulate the circuit and return the result'''
        stabilizer_measurement_cycles = random.randint(1, 300)
        stabilizer_measurement_cycles = 25
        circuit = seventeen_qubit_planar_code(stabilizer_measurement_cycles)
        
        # Execute the noisy simulation, measuring the stabilizers at each cycle
        result = execute(circuit, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=1).result()
        
        result = next(iter(result.get_counts()))

        result = [int(x) for x in result]
        # String is little endian, so C0S0 is the last bit. Reverse it.
        tensor_result = torch.tensor(result[::-1])
        # 1d tensor is ordered as [Cycle 1 Syndrome 1, Cycle 1 Syndrome 2, ... Cycle 1 Syndrome 7, Cycle 2 Syndrome 1, ...]
        tensor_result = tensor_result.reshape(stabilizer_measurement_cycles, -1)
        return tensor_result

if __name__ == '__main__':
    noise_model = get_noise(0.01, 0.01)
    ds = SurfaceCodeDataset(noise_model)
    print(ds[0])
