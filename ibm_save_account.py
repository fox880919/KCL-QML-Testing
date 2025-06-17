

from qiskit_ibm_runtime import QiskitRuntimeService


#36aJQeKjNGNtuKC12WQaVhmaBofxCGscdV2_Q5FtwSuR
old_ibm_token = "a2a37b0ae3d1e044e4a232ad054b7c52ee5613be17d0a48b0006b15e8e2090d02e3b1453cd07dc437529f352082c166b4eab60be49e5e5e777cab887f5aa8d36"

# new_ibm_token = "36aJQeKjNGNtuKC12WQaVhmaBofxCGscdV2_Q5FtwSuR"
# Replace 'YOUR_API_TOKEN' with your actual token from:
# https://quantum-computing.ibm.com/account
QiskitRuntimeService.save_account(channel="ibm_quantum", token=old_ibm_token, overwrite=True)
