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
#  Tests for Section B (Verifying the Doctest Examples)
# ==============================================================================

def test_example_1_easy():
    instance = Instance(
        agents={"p0"},
        items={"r1"},
        valuations={"p0": {"r1": 10}}
    )
    with pytest.raises(Exception, match="Empty implementation"):
        divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)

def test_example_2_simple():
    instance = Instance(
        agents={"p0", "p1"},
        items={"r1", "r2", "r3"},
        valuations={
            "p0": {"r1": 10, "r2": 5, "r3": 0},
            "p1": {"r1": 0, "r2": 5, "r3": 10}
        }
    )
    with pytest.raises(Exception, match="Empty implementation"):
        divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)

def test_example_3_collapse():
    """
    Verifies Example 3: Collapse scenario.
    """
    instance = Instance(
        agents={"p0", "p1"},
        items={"r1", "r2", "r3"},
        valuations={
            "p0": {"r1": 3, "r2": 3, "r3": 10},
            "p1": {"r1": 3, "r2": 3, "r3": 0}
        }
    )
    with pytest.raises(Exception, match="Empty implementation"):
        divide(lambda b: qp_local_search(b, T=15.0, epsilon=0.1), instance=instance)

# ==============================================================================
#  Tests for Section C (Additional Requirements)
# ==============================================================================

def test_edge_case_minimal_input():
    """
    Edge Case: Minimal valid instance (1 agent, 1 item with 0 value).
    Fixed to avoid fairpyx AssertionError by providing at least one item.
    """
    instance = Instance(valuations={"p0": {"r1": 0}})
    with pytest.raises(Exception, match="Empty implementation"):
        divide(lambda b: qp_local_search(b, T=15.0), instance=instance)

def test_large_input_fails():
    agents = {f"a{i}" for i in range(50)}
    items = {f"i{j}" for j in range(100)}
    valuations = {a: {i: 1 for i in items} for a in agents}
    instance = Instance(agents=agents, items=items, valuations=valuations)
    with pytest.raises(Exception, match="Empty implementation"):
        divide(lambda b: qp_local_search(b, T=1.0), instance=instance)

def test_random_input_fails():
    agents = {f"a{i}" for i in range(10)}
    items = {f"i{j}" for j in range(20)}
    valuations = {a: {i: random.randint(0, 10) for i in items} for a in agents}
    instance = Instance(agents=agents, items=items, valuations=valuations)
    with pytest.raises(Exception, match="Empty implementation"):
        divide(lambda b: qp_local_search(b, T=5.0), instance=instance)