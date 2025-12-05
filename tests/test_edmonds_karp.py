import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph import FlowNetwork
from edmonds_karp import edmonds_karp

# Create simple test network
network = FlowNetwork(4)
network.add_edge(0, 1, 10)  # s -> 1
network.add_edge(0, 2, 5)   # s -> 2
network.add_edge(1, 3, 10)  # 1 -> t
network.add_edge(2, 3, 5)   # 2 -> t

max_flow = edmonds_karp(network, source=0, sink=3)

print(f"Max flow: {max_flow}")
print(f"Expected: 15")
print(f"Test {'PASSED' if max_flow == 15 else 'FAILED'}")