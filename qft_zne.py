from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector, state_fidelity
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import LinearFactory, RichardsonFactory
import matplotlib.pyplot as plt
import os

# -----------------------------
# QFT circuit (3 qubits, no measurement - we need the quantum state itself)
# -----------------------------
def qft_circuit():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.append(QFT(num_qubits=3, do_swaps=True), [0, 1, 2])
    qc = transpile(qc, basis_gates=['h', 'x', 'cx', 'cp', 'swap', 'rz'], optimization_level=0)
    return qc

# -----------------------------
# Noise model (standardized: 5% depolarizing)
# -----------------------------
noise_model = NoiseModel()
error_1q = depolarizing_error(0.05, 1)
error_2q = depolarizing_error(0.05, 2)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cp', 'swap'])

# -----------------------------
# Ideal statevector (exact, no simulation needed)
# -----------------------------
qc = qft_circuit()
ideal_state = Statevector.from_instruction(qc)

# -----------------------------
# Executor: runs circuit, returns state fidelity against ideal statevector
# -----------------------------
def executor(circuit, noisy=True):
    qc = circuit.copy()
    qc.save_density_matrix()
    backend = AerSimulator(method='density_matrix', noise_model=noise_model if noisy else None)
    job = backend.run(qc)
    result = job.result()
    rho = result.data(0)['density_matrix']
    return state_fidelity(ideal_state, rho)

def noisy_executor(circuit):
    return executor(circuit, noisy=True)

# -----------------------------
# Run: Ideal vs Noisy vs ZNE-Mitigated (Linear and Richardson, more scale points)
# -----------------------------
ideal_value = 1.0
noisy_value = noisy_executor(qc)

mitigated_linear = zne.execute_with_zne(
    qc,
    noisy_executor,
    scale_noise=fold_global,
    factory=LinearFactory(scale_factors=[1, 2, 3, 4, 5])
)

mitigated_richardson = zne.execute_with_zne(
    qc,
    noisy_executor,
    scale_noise=fold_global,
    factory=RichardsonFactory(scale_factors=[1, 2, 3, 4, 5])
)

def report(name, value):
    error_before = abs(ideal_value - noisy_value)
    error_after = abs(ideal_value - value)
    reduction = (error_before - error_after) / error_before * 100 if error_before != 0 else 0
    print(f"\n{name}")
    print(f"  Mitigated fidelity: {value:.4f}")
    print(f"  Error before:       {error_before:.4f}")
    print(f"  Error after:        {error_after:.4f}")
    print(f"  Error reduction:    {reduction:.2f}%")
    return reduction

print("Ideal fidelity:", ideal_value)
print("Noisy fidelity:", noisy_value)

report("ZNE (Linear, 5 points)", mitigated_linear)
report("ZNE (Richardson, 5 points)", mitigated_richardson)

# -----------------------------
# Save comparison chart
# -----------------------------
os.makedirs("results", exist_ok=True)

labels = ["Ideal", "Noisy", "ZNE\n(Linear)", "ZNE\n(Richardson)"]
values = [ideal_value, noisy_value, mitigated_linear, mitigated_richardson]
colors = ["green", "red", "blue", "orange"]

plt.bar(labels, values, color=colors)
plt.ylabel("State Fidelity to Ideal")
plt.title("QFT: Ideal vs Noisy vs ZNE-Mitigated")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.savefig("results/qft_zne_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved chart to results/qft_zne_comparison.png")