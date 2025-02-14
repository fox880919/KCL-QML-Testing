import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import PassManager
# from qiskit.transpiler.passes import Unroller

# Create a sample quantum circuit
qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.cx(3, 4)

# qc.draw('mpl') 
# plt.show()


# Create a GenericBackendV2 instance
backend = GenericBackendV2(num_qubits=5) 

# Transpile the circuit with different optimization levels
transpiled_qc_level0 = transpile(qc, backend=backend, optimization_level=0)
transpiled_qc_level1 = transpile(qc, backend=backend, optimization_level=1)
transpiled_qc_level2 = transpile(qc, backend=backend, optimization_level=2)
transpiled_qc_level3 = transpile(qc, backend=backend, optimization_level=3)

transpiled_qc_level1.draw(output="mpl")
plt.show()

# Unroll the circuits to get basic gates (for easier comparison)
# pass_ = Unroller(['h', 'cx'])
# pm = PassManager(pass_)

# transpiled_qc_level0_unrolled = pm.run(transpiled_qc_level0)
# transpiled_qc_level1_unrolled = pm.run(transpiled_qc_level1)
# transpiled_qc_level2_unrolled = pm.run(transpiled_qc_level2)
# transpiled_qc_level3_unrolled = pm.run(transpiled_qc_level3)

# # Print the unrolled circuits 
# print("Optimization Level 0:")
# print(transpiled_qc_level0_unrolled) 
# print("\nOptimization Level 1:")
# print(transpiled_qc_level1_unrolled) 
# print("\nOptimization Level 2:")
# print(transpiled_qc_level2_unrolled) 
# print("\nOptimization Level 3:")
# print(transpiled_qc_level3_unrolled) 