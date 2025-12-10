"""
linear_programming.py - Linear Programming Formulation for Maximum Flow

Solves maximum flow using Gurobi optimizer with explicit LP formulation.
"""

import gurobipy as gp
from gurobipy import GRB
from graph import FlowNetwork

def max_flow_lp(network, source, sink):
    """
    Solve maximum flow using Linear Programming (Gurobi).
    
    Args:
        network: FlowNetwork object
        source: source node index
        sink: sink node index
    
    Returns:
        Maximum flow value
    """
    # Create model
    model = gp.Model("MaxFlow")
    model.setParam('OutputFlag', 0)
    
    # Create decision variables for each edge (loop through all edges of all nodes in our network)
    flow_variables = {}

    for u in range(network.n):
        for edge in network.graph[u]:
            v = edge.dest
            capacity = edge.capacity

            # Add each ensuring capacity constraint
            flow_variables[(u,v)] = model.addVar(lb=0.0, ub=capacity, name=f"f_{u}_{v}")

    # Set objective function (Maximize Σ flow[source, v] for all v where there's an edge from source to v)
    model.setObjective(
        gp.quicksum(flow_variables[(source, edge.dest)] for edge in network.graph[source]),
        GRB.MAXIMIZE
    )

    # Add Flow Conservation constraint (flow in = flow out) for all nodes except source and sink
    for node in range(network.n):
        if node == source or node == sink:
            continue
        
        # Calculate flow IN to this node
        flow_in = gp.quicksum(
            flow_variables[(u, node)] 
            for u in range(network.n) 
            if (u, node) in flow_variables
        )
        
        # Calculate flow OUT from this node
        flow_out = gp.quicksum(
            flow_variables[(node, edge.dest)] 
            for edge in network.graph[node]
        )
        
        # Add constraint: flow_in == flow_out
        model.addConstr(flow_in == flow_out, name=f"conservation_{node}")

    # Solve and return result
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        print("Optimization failed")
        return 0