
import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()

#write to file
qasm_str = dump(qc, 'qasm_code.qasm')

#read from qasm file
qc = QuantumCircuit.from_qasm_file('qasm_code.qasm')

qc.draw(output="mpl")
plt.show()



