"""
Unit tests for the Quasi-Polynomial Local Search algorithm.
Fulfills Section C of the assignment.

Programmer: Rotem Melamed
Date: 08/11/2025
"""

import pytest
import random
from fairpyx import Instance, divide
from fairpyx.algorithms.qp_local_search import qp_local_search

# ==============================================================================
#  Tests
# ==============================================================================

def test_example_1_easy():
    """
    Verifies Example 1: Single agent and single item.
    """
    instance = Instance(
        agents={"p0"},
        items={"r1"},
        valuations={"p0": {"r1": 10}}
    )
    # We expect {'p0': {'r1'}}
    result = divide(qp_local_search,  instance, T=15.0, epsilon=0.1)
    assert result == {'p0': {'r1'}}

def test_example_2_simple():
    """
    Verifies Example 2: Simple example with two agents and three items.
    """
    instance = Instance(
        agents={"p0", "p1"},
        items={"r1", "r2", "r3"},
        valuations={
            "p0": {"r1": 10, "r2": 5, "r3": 0},
            "p1": {"r1": 0, "r2": 5, "r3": 10}
        }
    )
    # We expect {p0: {'r1'}, p1: {'r3'}}
    result = divide(qp_local_search,  instance, T=15.0, epsilon=0.1)
    assert result == {'p0': {'r1'}, 'p1': {'r3'}}

def test_example_3_collapse():
    """
    Verifies Example 3: Collapse scenario.
    p0 should get r3(fat item), and p1 should get r1 and r2(thin items where they both above threshold of 3.65=15/4.1).
    """
    instance = Instance(
        agents={"p0", "p1"},
        items={"r1", "r2", "r3"},
        valuations={
            "p0": {"r1": 3, "r2": 3, "r3": 10},
            "p1": {"r1": 3, "r2": 3, "r3": 0}
        }
    )
    # We expect {p0: {'r3'}, p1: {'r1', 'r2'}}
    result = divide(qp_local_search,  instance, T=15.0, epsilon=0.1)
    assert result == {'p0': {'r3'}, 'p1': {'r1', 'r2'}}

# ==============================================================================
#  Edge Cases
# ==============================================================================

def test_edge_case_minimal_input():
    """
    Edge Case: Minimal valid instance (1 agent, 1 item with 0 value).
    Should result in the agent receiving no items.
    """
    instance = Instance(valuations={"p0": {"r1": 0}})
    # This might fail with Exception in empty implementation
    result = divide(qp_local_search,  instance, T=15.0, epsilon=0.1)
    assert result == {"p0": set()}

def test_large_input():
    """
    Edge Case: Large instance with many agents and items.
    Uses random valuations with a higher chance for zero values.
    """
    # We set a seed for reproducibility
    seed = random.randint(0, 10**9)
    print(f"Test seed: {seed}")
    random.seed(seed)

    # We create 10 agents and 20 items, increasing indexes
    agents = []
    items = []
    numagents = 10
    numitems = 20
    for i in range(numagents):
        agents.append(f"a{i}")
    for j in range(numitems):
        items.append(f"i{j}")

    #Random valuations between 0 and 10, the value for each item is equal for all agents, with higher chance for 0
    valuations = {}
    for agent in agents:
        valuations[agent] = {}
        for item in items:
            valuations[agent][item] = random.choices([0, random.randint(1, 20)], weights=[0.6, 0.4])[0]
        print("agent", agent, "valuations:", valuations[agent])
    
    T = 2

    instance = Instance(agents=agents, items=items, valuations=valuations)

    result = divide(qp_local_search,  instance, T=15.0, epsilon=0.1)

    # We just check that a dict is returned
    assert isinstance(result, dict)

    # We check that there are no duplicate items assigned
    all_allocated_items = set()
    for agent, bundle in result.items():
        # Check that agent is valid
        assert agent in agents, f"Unknown agent {agent} in result!"

        # Calculate agent value
        agent_value = sum(instance.agent_item_value(agent, item) for item in bundle)

        # Check no item is assigned to multiple agents
        for item in bundle:
            assert item not in all_allocated_items, f"Item {item} assigned to multiple agents!"
            all_allocated_items.add(item)
    
    # We check that all allocated items are from the original set
    for item in all_allocated_items:
        assert item in items, f"Unknown item {item} in allocation!"

    # We check that every agent meets the target T
    for agent, bundle in result.items():
        agent_value = sum(instance.agent_item_value(agent, item) for item in bundle)
        assert agent_value >= T, f"Agent {agent} has value {agent_value} which is below target {T}!"