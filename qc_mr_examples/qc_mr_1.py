import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(3)

theta = np.pi/2  # Define the rotation angle (in radians)

# qc.rx(theta, 0)  
# qc.x(1)
# qc.h(2)
# qc.cx(1, 2)

# #show 1
# qc.draw(output="mpl")
# plt.show()

qc.rx(theta, 2)  
qc.x(0)
qc.h(1)
qc.cx(0, 1)

#show 1
qc.draw(output="mpl")
plt.show()



