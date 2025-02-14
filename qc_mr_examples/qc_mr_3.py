import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit

qc = QuantumCircuit(4)

qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)

#show 1
qc.draw(output="mpl")
plt.show()

#or show 2
#print(qc.draw())