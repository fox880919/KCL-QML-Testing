import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
import numpy as np

theta = np.pi  # Define the rotation angle (in radians)


# qc = QuantumCircuit(4)

# qc.x(0)
# qc.h(1)
# qc.cx(1, 0)
# qc.cx(2, 3)
# qc.z(3)


# #show 1
# qc.draw(output="mpl")
# plt.show()


# qc = QuantumCircuit(2)

# qc.x(0)
# qc.h(1)

# #show 1
# qc.draw(output="mpl")
# plt.show()

qc = QuantumCircuit(2)

qc.cx(0, 1)
qc.z(1)
#show 1
qc.draw(output="mpl")
plt.show()