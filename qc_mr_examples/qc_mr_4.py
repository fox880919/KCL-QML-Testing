import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
import numpy as np

theta = np.pi  # Define the rotation angle (in radians)


# qc = QuantumCircuit(2)

# qc.x(0)
# qc.h(0)
# qc.cx(0, 1)

# #show 1
# qc.draw(output="mpl")
# plt.show()


qc = QuantumCircuit(2)

qc.rx(theta, 0)  
qc.h(0)
qc.cx(0, 1)

#show 1
qc.draw(output="mpl")
plt.show()


#or show 2
#print(qc.draw())