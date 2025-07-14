

from qiskit_ibm_runtime import QiskitRuntimeService

import os

from dotenv import load_dotenv
load_dotenv()
ibm_api_token = os.environ.get('IBM_API_TOKEN')
print(f'IBM_API_TOKEN: {ibm_api_token}')

# CRN = os.environ.get('CRN')
# print(f'CRN: {CRN}')

from qiskit_ibm_runtime import QiskitRuntimeService

# QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_api_token, overwrite=True)

service = QiskitRuntimeService(instance='faiez_first_free_instance')

print(f'service: {service}')

# backends = service.backend()

# backend = service.backend("ibm_brisbane")

# print(f'backend: {backend}')
