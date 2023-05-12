import numpy as np

with open("results/neural_network_results.npy", "rb") as f:
    a = np.load(f)
    b = np.load(f)
    c = np.load(f)
    d = np.load(f)

print(a,b,c,d)