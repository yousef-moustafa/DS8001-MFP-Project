import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph import FlowNetwork
from edmonds_karp import edmonds_karp
from linear_programming import max_flow_lp

# Create test network
network1 = FlowNetwork(4)
network1.add_edge(0, 1, 10)
network1.add_edge(0, 2, 5)
network1.add_edge(1, 3, 10)
network1.add_edge(2, 3, 5)

# Test Edmonds-Karp
start_time = time.time()
ek_flow = edmonds_karp(network1, 0, 3)
ek_time = time.time() - start_time

# Test LP (need fresh network since EK modifies it)
network2 = FlowNetwork(4)
network2.add_edge(0, 1, 10)
network2.add_edge(0, 2, 5)
network2.add_edge(1, 3, 10)
network2.add_edge(2, 3, 5)

start_time = time.time()
lp_flow = max_flow_lp(network2, 0, 3)
lp_time = time.time() - start_time

print(f"Edmonds-Karp: {ek_flow} (Time: {ek_time:.6f}s)")
print(f"LP (Gurobi):  {lp_flow} (Time: {lp_time:.6f}s)")
print(f"Match: {ek_flow == lp_flow}")