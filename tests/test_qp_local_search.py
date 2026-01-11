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
    result = divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)
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
    result = divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)
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
    result = divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)
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
    result = divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)
    assert result == {"p0": set()}

def test_large_input():
    """
    Edge Case: Large instance with many agents and items.
    The valuations should be:
    agent a0 valuations: {'i0': 0, 'i1': 0, 'i2': 3, 'i3': 0, 'i4': 0, 'i5': 0, 'i6': 9, 'i7': 0, 'i8': 0, 'i9': 8}
    agent a1 valuations: {'i0': 1, 'i1': 3, 'i2': 0, 'i3': 4, 'i4': 0, 'i5': 0, 'i6': 6, 'i7': 1, 'i8': 0, 'i9': 0}
    agent a2 valuations: {'i0': 5, 'i1': 10, 'i2': 6, 'i3': 0, 'i4': 4, 'i5': 2, 'i6': 0, 'i7': 8, 'i8': 0, 'i9': 0}
    agent a3 valuations: {'i0': 5, 'i1': 2, 'i2': 3, 'i3': 0, 'i4': 0, 'i5': 0, 'i6': 6, 'i7': 0, 'i8': 1, 'i9': 0}
    agent a4 valuations: {'i0': 4, 'i1': 10, 'i2': 0, 'i3': 0, 'i4': 0, 'i5': 0, 'i6': 9, 'i7': 0, 'i8': 0, 'i9': 4}
    agent a5 valuations: {'i0': 3, 'i1': 2, 'i2': 0, 'i3': 3, 'i4': 7, 'i5': 0, 'i6': 8, 'i7': 9, 'i8': 1, 'i9': 2}
    agent a6 valuations: {'i0': 9, 'i1': 0, 'i2': 0, 'i3': 1, 'i4': 5, 'i5': 3, 'i6': 2, 'i7': 5, 'i8': 9, 'i9': 0}
    agent a7 valuations: {'i0': 3, 'i1': 9, 'i2': 0, 'i3': 0, 'i4': 6, 'i5': 0, 'i6': 4, 'i7': 0, 'i8': 8, 'i9': 9}
    agent a8 valuations: {'i0': 3, 'i1': 0, 'i2': 9, 'i3': 7, 'i4': 9, 'i5': 4, 'i6': 7, 'i7': 0, 'i8': 0, 'i9': 0}
    agent a9 valuations: {'i0': 0, 'i1': 0, 'i2': 0, 'i3': 0, 'i4': 1, 'i5': 2, 'i6': 5, 'i7': 4, 'i8': 10, 'i9': 4}
    """
    # We set a seed for reproducibility
    random.seed(42)
    agents = []
    items = []
    for i in range(10):
        agents.append(f"a{i}")
    for j in range(10):
        items.append(f"i{j}")

    #Random valuations between 0 and 10, the value for each item is equal for all agents, with higher chance for 0
    valuations = {}
    for agent in agents:
        valuations[agent] = {}
        for item in items:
            valuations[agent][item] = random.choices([0, random.randint(1, 10)], weights=[0.5, 0.5])[0]
        print("agent", agent, "valuations:", valuations[agent])
    instance = Instance(agents=agents, items=items, valuations=valuations)

    result = divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)
    assert isinstance(result, dict)