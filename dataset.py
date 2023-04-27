import os
import pandas as pd
from torch.utils.data import Dataset
import random
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from simulate import seventeen_qubit_planar_code, get_noise

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
        stabilizer_measurement_cycles = 2
        circuit = seventeen_qubit_planar_code(stabilizer_measurement_cycles)
        
        # Execute the noisy simulation, measuring the stabilizers at each cycle
        result = execute(circuit, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=1).result()
        return result.get_counts()

if __name__ == '__main__':
    noise_model = get_noise(0.01, 0.01)
    ds = SurfaceCodeDataset(noise_model)
