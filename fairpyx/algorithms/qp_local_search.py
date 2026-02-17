"""
An implementation of the algorithms in:
"Quasi-Polynomial Local Search for Restricted Max-Min Fair Allocation"
By Lukas Polacek and Ola Svensson (2014)
[https://arxiv.org/pdf/1205.1373](https://arxiv.org/pdf/1205.1373)

Programmer: Rotem Melamed
Date: 10/02/2026
"""
from fairpyx.allocations import AllocationBuilder
from fairpyx.instances import Instance
from fairpyx import divide
from typing import Set, Dict, List, Optional, Tuple, Any
import math
import logging
import pulp
import numpy as np

orig_sorted = AllocationBuilder.sorted

def patched_sorted(self):
    bundles = orig_sorted(self)
    return {agent: set(items) for agent, items in bundles.items()}

AllocationBuilder.sorted = patched_sorted


# --- Logging Setup ---
logger = logging.getLogger("QP_Local_Search")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --- LP Configuration ---
def knapsack_separation_oracle(agent: Any, instance: Instance, T: float, z_values: Dict[Any, float]) -> Optional[Tuple[List[Any], float]]:
    """
    Knapsack-based separation oracle for dual constraint:
    For agent i, find min { sum_{j in C} z_j : C subseteq R, sum_{j in C} v_ij >= T }

    Returns (bundle, min_cost) if a bundle exists, or None if infeasible.
    Uses dynamic programming with values v_ij and costs z_j.
    """
    items = list(instance.items)
    n = len(items)

    # Get agent-specific values
    values = [instance.agent_item_value(agent, item) for item in items]
    costs = [z_values.get(item, 0.0) for item in items]

    # DP: dp[v] = minimum cost to achieve total value >= v
    # We discretize values to avoid floating point issues
    scale = 1000  # Scale factor for discretization
    T_scaled = int(T * scale)
    values_scaled = [int(v * scale) for v in values]

    max_value = sum(values_scaled)
    if max_value < T_scaled:
        return None  # Infeasible

    # Initialize DP table
    INF = float('inf')
    dp = [INF] * (max_value + 1) # dp[v] = min cost to achieve value >= v
    dp[0] = 0.0
    parent = [None] * (max_value + 1)  # Parent[v] = (item_index, previous_value) that led to dp[v]

    # DP computation
    for idx in range(n):
        v = values_scaled[idx]
        c = costs[idx]

        # Traverse backwards to avoid using same item multiple times
        for val in range(max_value, v - 1, -1):
            if dp[val - v] + c < dp[val]:
                dp[val] = dp[val - v] + c
                parent[val] = (idx, val - v)  # (item_index, previous_value)

    # Find minimum cost for value >= T
    min_cost = INF
    best_val = None
    for val in range(T_scaled, max_value + 1):
        if dp[val] < min_cost:
            min_cost = dp[val]
            best_val = val

    if min_cost == INF:
        return None

    # Reconstruct the bundle
    bundle = []
    used_indices = set()
    current_val = best_val
    while parent[current_val] is not None:
        idx, prev_val = parent[current_val]
        if idx not in used_indices:
            bundle.append(items[idx])
            used_indices.add(idx)
        current_val = prev_val

    return (bundle, min_cost)


def solve_configuration_lp(instance: Instance, T: float, epsilon: float = 0.1) -> Dict[Any, List[tuple]]:
    """
    Solves the configuration LP dual using cutting-plane method.

    Dual LP:
    - Variables: y_i >= 0 for each player, z_j >= 0 for each item
    - Objective: maximize sum_i y_i - sum_j z_j
    - Constraints: for all i and C in C(i,T), y_i <= sum_{j in C} z_j

    Uses knapsack separation oracle to generate violated constraints iteratively.

    Returns: configs_per_agent - dict mapping each agent to list of minimal configurations
    """
    agents = list(instance.agents)
    items = list(instance.items)
    alpha = 4 + epsilon
    threshold = T / alpha

    logger.info(f"\n=== Solving Configuration LP for T={T:.2f}, threshold={threshold:.2f} ===")

    # Initialize LP problem
    prob = pulp.LpProblem("ConfigLP_Dual", pulp.LpMaximize)

    # Calculate upper bounds: max total value any agent can get
    max_agent_value = {}
    max_item_value = {}
    for agent in agents:
        max_agent_value[agent] = sum(instance.agent_item_value(agent, item) for item in items)
    for item in items:
        max_item_value[item] = max(instance.agent_item_value(agent, item) for agent in agents)

    # Variables with reasonable upper bounds to prevent unboundedness
    y_vars = {agent: pulp.LpVariable(f"y_{agent}", lowBound=0, upBound=max_agent_value[agent]) for agent in agents}
    z_vars = {item: pulp.LpVariable(f"z_{item}", lowBound=0, upBound=max_item_value[item]) for item in items}

    # Objective: maximize sum(y) - sum(z)
    prob += pulp.lpSum(y_vars.values()) - pulp.lpSum(z_vars.values())

    # Track configurations added
    configs_per_agent = {agent: [] for agent in agents}
    constraint_count = 0

    # Iterative constraint generation
    max_iterations = 1000
    for iteration in range(max_iterations):
        # Solve current LP
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.LpStatusOptimal:
            logger.warning(f"LP not optimal at iteration {iteration}, status: {pulp.LpStatus[prob.status]}")
            break

        # Get current dual values
        y_current = {agent: y_vars[agent].varValue for agent in agents}
        z_current = {item: z_vars[item].varValue for item in items}

        logger.debug(f"Iteration {iteration}: obj = {pulp.value(prob.objective):.4f}")

        # Check for violated constraints using separation oracle
        violations_found = 0

        for agent in agents:
            result = knapsack_separation_oracle(agent, instance, threshold, z_current)

            if result is None:
                # Agent cannot achieve threshold - this means LP is infeasible for this T
                logger.warning(f"Agent {agent} cannot achieve threshold {threshold:.2f}")
                continue

            bundle, min_cost = result

            # Check if constraint is violated: y_i > sum_{j in C} z_j
            violation = y_current[agent] - min_cost

            if violation > 1e-6:  # Tolerance for numerical errors
                # Add violated constraint (deduplicate items)
                bundle_tuple = tuple(sorted(set(bundle)))
                constraint_name = f"config_{agent}_{constraint_count}"
                prob += y_vars[agent] <= pulp.lpSum(z_vars[item] for item in bundle_tuple), constraint_name

                # Store configuration
                configs_per_agent[agent].append(bundle_tuple)
                constraint_count += 1
                violations_found += 1

                logger.debug(f"  Violated: {agent}, bundle size {len(bundle)}, violation {violation:.4f}")

        if violations_found == 0:
            logger.info(f"No violations found. LP solved in {iteration + 1} iterations with {constraint_count} constraints.")
            break

        logger.debug(f"  Added {violations_found} constraints")

    # Filter to minimal configurations AND ensure all minimal single-item configs are included
    for agent in agents:
        configs = configs_per_agent[agent]

        # Ensure all valid single-item configurations are included
        for item in items:
            item_value = instance.agent_item_value(agent, item)
            if item_value >= threshold:
                single_config = (item,)
                if single_config not in configs:
                    configs.append(single_config)

        minimal_configs = []

        for config in configs:
            # Check if config is minimal (no proper subset also in configs)
            is_minimal = True
            config_set = set(config)

            for other_config in configs:
                if other_config == config:
                    continue
                other_set = set(other_config)
                if other_set < config_set:  # Proper subset
                    # Check if subset achieves threshold
                    other_value = sum(instance.agent_item_value(agent, item) for item in other_config)
                    if other_value >= threshold:
                        is_minimal = False
                        break

            if is_minimal:
                minimal_configs.append(config)

        # Remove duplicates
        configs_per_agent[agent] = list(set(minimal_configs))
        logger.info(f"Agent {agent}: {len(configs_per_agent[agent])} minimal configurations")

    return configs_per_agent


# --- Helper Class ---
class AlternatingTree:
    """
    A class to represent an alternating tree.
    Manages sets A (addable) and B (blocking) and supports Pruning.
    """
    def __init__(self, root_player: Any, T: float, alpha: float, precomputed_configs: Optional[Dict[Any, List[tuple]]] = None):
        self.root = root_player
        self.T = T
        self.alpha = alpha
        self.precomputed_configs = precomputed_configs  # LP-generated configurations

        # Key: tuple(items) -> meta
        # meta keys: items, player, distance, type, layer, parent
        # note- parent -> key of the preceding edge on the alternating path (or None for root A-edge)

        self.A_edges: Dict[tuple, dict] = {}
        self.B_edges: Dict[tuple, dict] = {}

        self.visited_bundles = set()
    
    def get_distance(self, bundle_items: tuple, player: Any, instance: Instance) -> int:
        """Determines fat/thin classification: 0 for fat (single item >= threshold), 1 for thin."""
        threshold = self.T / self.alpha
        if len(bundle_items) == 1:
            val = instance.agent_item_value(player, bundle_items[0])
            if val >= threshold:
                return 0  # Fat edge
        return 1  # Thin edge
    
    def _get_minimal_valid_bundles(self, agent: Any, instance: Instance) -> List[tuple]:
        """
        Returns the minimal configurations C in C(i, T/alpha) from the LP oracle.

        These configurations are generated by the configuration LP dual solver,
        ensuring quasi-polynomial complexity.
        """
        if self.precomputed_configs is None:
            logger.error("No precomputed configs provided! LP oracle must be run first.")
            return []

        configs = self.precomputed_configs.get(agent, [])
        logger.debug(f"Using {len(configs)} LP-generated configs for {agent}")
        return configs
    
    def find_addable_edge(self, instance, max_distance) -> Optional[dict]:
        best_edge = None
        min_dist = float('inf')
        best_is_root = False  # Prefer root player edges (they collapse directly)

        # Candidates: root player + all players in the tree (A/B edges)
        candidate_players = {self.root}
        for meta in self.A_edges.values():
            candidate_players.add(meta['player'])
        for meta in self.B_edges.values():
            candidate_players.add(meta['player'])

        tree_items = set()
        for meta in self.A_edges.values():
            tree_items.update(meta['items'])
        for meta in self.B_edges.values():
            tree_items.update(meta['items'])

        logger.debug(f"  > Candidates: {len(candidate_players)}, tree_items: {len(tree_items)}, max_dist: {max_distance}")

        for player in candidate_players:
            # Parent distance
            parent_dist = 0
            if player != self.root:
                found = False
                for b_meta in self.B_edges.values():
                    if b_meta['player'] == player:
                        parent_dist = b_meta['distance']
                        found = True
                        break
                if not found:
                    for a_meta in self.A_edges.values():
                        if a_meta['player'] == player:
                            parent_dist = a_meta['distance']
                            found = True
                            break
                if not found:
                    continue

            if parent_dist >= max_distance:
                continue

            potential_bundles = self._get_minimal_valid_bundles(player, instance)
            logger.debug(f"    Player {player}: {len(potential_bundles)} bundles")

            for bundle in potential_bundles:
                if bundle in self.visited_bundles:
                    continue
                if not set(bundle).isdisjoint(tree_items):
                    continue

                edge_cost = self.get_distance(bundle, player, instance)
                total_dist = parent_dist + edge_cost

                if total_dist <= max_distance:
                    is_root = (player == self.root)
                    # Prefer root player edges (they collapse directly),
                    # then prefer lower distance
                    if (is_root and not best_is_root) or \
                       (is_root == best_is_root and total_dist < min_dist):
                        min_dist = total_dist
                        best_is_root = is_root
                        if edge_cost == 0:
                            edge_type = 'fat'
                            layer = 2* (total_dist // 2)  # Even layer for fat
                        else:
                            edge_type = 'thin'
                            layer = 2 * (total_dist // 2) + 1  # Odd layer for thin
                        best_edge = {
                            'items': bundle,
                            'player': player,
                            'distance': total_dist,
                            'type': edge_type,
                            'layer': layer,
                            'parent': None  # To be set when added to tree
                        }
        return best_edge
    
    def get_blocking_edges(self, edge_to_add: dict, matching: Dict[Any, Set]) -> List[dict]:
        blockers = []
        new_items = set(edge_to_add['items'])
        add_layer = edge_to_add.get('layer', edge_to_add['distance'])

        for m_player, m_bundle in matching.items():
            if not m_bundle:
                continue
            if new_items.isdisjoint(m_bundle):
                continue

            b_items = tuple(sorted(m_bundle))

            # B-edge shares the same layer as the A-edge it blocks.
            # Parent of a B-edge is the A-edge it blocks.
            blocker = {
                'items': b_items,
                'player': m_player,
                'distance': edge_to_add['distance'],
                'layer': add_layer,
                'blocking_who': edge_to_add['items'],
                'parent': tuple(sorted(edge_to_add['items'])),
            }
            blockers.append(blocker)

        return blockers


    def prune(self, max_allowed_distance: int):
        """
        Removes all A-edges and B-edges with distance exceeding max_allowed_distance.
        """
        # Remove deep A edges
        keys_to_remove_A = [k for k, v in self.A_edges.items() if v['distance'] > max_allowed_distance]
        for k in keys_to_remove_A:
            del self.A_edges[k]
            self.visited_bundles.discard(k)  # Use discard to avoid KeyError
        
        # Remove deep B edges + B blocking removed A
        keys_to_remove_B = [k for k, v in self.B_edges.items() if v['distance'] > max_allowed_distance]
        for k in keys_to_remove_B:
            del self.B_edges[k]
        
        logger.info(f"    [Pruning] >{max_allowed_distance}. A-rem: {len(keys_to_remove_A)}, B-rem: {len(keys_to_remove_B)}")


def safe_swap(alloc, player_losing, items_losing, player_getting, items_getting):
    """Safely swap bundles."""
    # Remove old
    for item in items_losing:
        if item in alloc.bundles[player_losing]:
            alloc.bundles[player_losing].remove(item)
        if hasattr(alloc, 'remaining_item_capacities'):
            alloc.remaining_item_capacities[item] = alloc.remaining_item_capacities.get(item, 0) + 1
            
    # Assign new
    for item in items_getting:
        alloc.give(player_getting, item)


# --- Main Algorithm (Algorithm 1) ---
def qp_local_search(alloc: AllocationBuilder, T: float, epsilon: float = 0.1, precomputed_configs: Dict[Any, List[tuple]] = None) -> bool:
    """
    QP local search for restricted max-min fair allocation.

    This is the core local search algorithm from the paper, using LP-generated
    configurations to achieve quasi-polynomial complexity.

    Args:
        alloc: AllocationBuilder with instance
        T: Target value
        epsilon: Approximation parameter
        precomputed_configs: LP-generated configurations (required)

    Returns:
        True if all agents satisfied, False otherwise
    """
    if precomputed_configs is None:
        raise ValueError("precomputed_configs is required. Use qp_local_search_with_lp() to run the full algorithm.")

    instance = alloc.instance
    alpha = 4 + epsilon
    threshold = T / alpha

    logger.info(f"=== QP Local Search: T={T:.2f}, thresh={threshold:.2f}, α={alpha:.2f} ===")
    logger.info(f"Using configuration oracle")

    def is_satisfied(agent):
        return sum(instance.agent_item_value(agent, item) for item in alloc.bundles.get(agent, [])) >= threshold

    unsatisfied_agents = [p for p in instance.agents if not is_satisfied(p)]

    while unsatisfied_agents:
        root_agent = unsatisfied_agents[0]
        logger.info(f"\nSatisfy {root_agent} (unsat: {len(unsatisfied_agents)})")

        alt_tree = AlternatingTree(root_agent, T, alpha, precomputed_configs)
        
        
        num_players = len(instance.agents)
        base = 1 + epsilon / 3
        max_dist = 2 * math.ceil(math.log(num_players, base)) + 1 if num_players > 0 else 1

        
        logger.info(f"  max_dist={max_dist}")
        
        path_found = False
        
        while True:
            # 1. Find minimum-distance addable edge
            addable_edge = alt_tree.find_addable_edge(instance, max_dist)
            
            if addable_edge is None:
                logger.warning(f"  No addable edge <= dist {max_dist}")
                break 
            
            e_key = tuple(sorted(addable_edge['items']))

            # Determine parent edge key for this player in the tree, if any.
            parent_key = None
            for k, b in alt_tree.B_edges.items():
                if b['player'] == addable_edge['player']:
                    parent_key = k
                    break
            if parent_key is None:
                for k, a in alt_tree.A_edges.items():
                    if a['player'] == addable_edge['player']:
                        parent_key = k
                        break

            addable_edge['parent'] = parent_key

            alt_tree.A_edges[e_key] = addable_edge
            alt_tree.visited_bundles.add(e_key)

            
            # 2. Blockers from current alloc
            current_matching = {a: s for a, s in alloc.bundles.items() if s}
            blocking_edges = alt_tree.get_blocking_edges(addable_edge, current_matching)
            
            if blocking_edges:
                for b in blocking_edges:
                    b_key = tuple(sorted(b['items']))
                    alt_tree.B_edges[b_key] = b
                    logger.info(f"  BLOCKER: {b_key} ({b['player']})")
            
            else:
                # 3. COLLAPSE (Algorithm 1)
                logger.info(f"  FREE {e_key} → COLLAPSE")

                # Check if this edge is for the root agent or properly connected via B-edges
                edge_player = addable_edge['player']
                if edge_player != root_agent:
                    # For non-root players, they must have a blocker in B_edges to be connected
                    has_blocker = any(b['player'] == edge_player for b in alt_tree.B_edges.values())
                    if not has_blocker:
                        logger.warning("  Collapse failed: non-root player not in tree")
                        continue

                curr_key = e_key
                last_removed_blocker = None
                collapse_succeeded = False

                while True:
                    curr_e = alt_tree.A_edges[curr_key]

                    # Check if any B still blocks curr_e
                    if any(
                        not set(curr_e['items']).isdisjoint(b_edge['items'])
                        for b_edge in alt_tree.B_edges.values()
                    ):
                        break  # Partial collapse → prune

                    player = curr_e['player']

                    if player == root_agent:
                        logger.info(f"    ROOT {player} ← {curr_e['items']}")
                        for item in curr_e['items']:
                            alloc.give(player, item)
                        collapse_succeeded = True
                        break

                    # There should be exactly one B-edge in the tree covering this player
                    e_prime_key = None
                    e_prime = None
                    for k, b in alt_tree.B_edges.items():
                        if b['player'] == player and not set(curr_e['items']).isdisjoint(b['items']):
                            e_prime_key, e_prime = k, b
                            break

                    if e_prime is None:
                        # No blocker for this player in B, something inconsistent
                        logger.warning("  Collapse failed: no blocker in tree")
                        break

                    logger.info(f"    SWAP {player}: {e_prime['items']} → {curr_e['items']}")
                    safe_swap(alloc, player, e_prime['items'], player, curr_e['items'])

                    # Remove this blocker from B
                    del alt_tree.B_edges[e_prime_key]
                    last_removed_blocker = e_prime

                    # Move to its parent A-edge on the tree
                    parent_key = e_prime.get('parent')
                    if parent_key is None or parent_key not in alt_tree.A_edges:
                        # Should only happen at the root predecessor
                        break
                    curr_key = parent_key


                if collapse_succeeded:
                    path_found = True
                    break
                else:
                    # Prune to the distance of the last removed blocker
                    if last_removed_blocker:
                        dist = last_removed_blocker['distance']
                        alt_tree.prune(dist)
                    else:
                        logger.warning("  Collapse failed, no blocker?")

        if path_found:
            unsatisfied_agents = [p for p in instance.agents if not is_satisfied(p)]
            logger.info(f"  SUCCESS. Remaining: {len(unsatisfied_agents)}")
        else:
            # If we never found any path for this root, just stop.
            # divide() will return the current (possibly empty) allocation.
            logger.info(f"  No augmenting path for {root_agent}; leaving allocation as is.")
            return False
    
    logger.info("=== ALL AGENTS SATISFIED ===")
    return True


# --- Quasi-Polynomial Algorithm with LP Oracle ---
def qp_local_search_with_lp(alloc: AllocationBuilder, T: float, epsilon: float = 0.1) -> bool:
    """
    Quasi-polynomial algorithm using LP-based configuration oracle.

    This achieves the true quasi-poly time complexity from the paper by:
    1. Solving the configuration LP dual with knapsack separation oracle
    2. Using only LP-generated configurations in the local search

    Args:
        alloc: AllocationBuilder with instance
        T: Target value
        epsilon: Approximation parameter

    Returns:
        True if all agents can achieve T/(4+epsilon), False otherwise
    """
    instance = alloc.instance

    # Step 1: Solve configuration LP to get configurations
    logger.info(f"\n{'='*60}")
    logger.info(f"QUASI-POLYNOMIAL ALGORITHM (LP-based)")
    logger.info(f"{'='*60}")

    try:
        precomputed_configs = solve_configuration_lp(instance, T, epsilon)
    except Exception as e:
        logger.error(f"LP solving failed: {e}")
        return False

    # Step 2: Run local search with LP-generated configurations
    return qp_local_search(alloc, T, epsilon, precomputed_configs)


def qp_max_min_allocation(instance: Instance, epsilon: float = 0.1) -> Dict[Any, Set]:
    """
    Find maximum T such that all agents can achieve value >= T/(4+epsilon).

    Uses binary search on T with LP-based configuration oracle to achieve
    quasi-polynomial time complexity O(n^((1/ε)log n)).

    Args:
        instance: Instance with agents, items, and valuations
        epsilon: Approximation parameter (default 0.1 for 1/(4.1)-approximation)

    Returns:
        Allocation dict mapping agents to sets of items
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"QUASI-POLYNOMIAL MAX-MIN ALLOCATION")
    logger.info(f"Epsilon: {epsilon}, Approximation ratio: 1/{4+epsilon:.2f}")
    logger.info(f"Complexity: O(n^((1/ε)log n))")
    logger.info(f"{'='*60}\n")

    # Determine search range for T
    # Lower bound: 0
    # Upper bound: maximum possible value for any agent
    max_total_value = 0
    for agent in instance.agents:
        total_value = sum(instance.agent_item_value(agent, item) for item in instance.items)
        max_total_value = max(max_total_value, total_value)

    if max_total_value == 0:
        logger.warning("All items have zero value for all agents")
        return {agent: set() for agent in instance.agents}

    # Binary search on T
    T_low = 0.0
    T_high = max_total_value
    best_T = 0.0
    best_allocation = None
    tolerance = 1e-3

    iteration = 0
    max_iterations = int(math.log2(max_total_value / tolerance)) + 10

    while T_high - T_low > tolerance and iteration < max_iterations:
        T_mid = (T_low + T_high) / 2
        iteration += 1

        logger.info(f"\n--- Binary Search Iteration {iteration} ---")
        logger.info(f"Range: [{T_low:.4f}, {T_high:.4f}], Testing T = {T_mid:.4f}")

        # Create fresh allocation
        from fairpyx.allocations import AllocationBuilder
        alloc = AllocationBuilder(instance)

        # Try to achieve T_mid using LP oracle
        success = qp_local_search_with_lp(alloc, T_mid, epsilon)

        if success:
            # All agents satisfied - try higher T
            best_T = T_mid
            best_allocation = alloc.sorted()
            T_low = T_mid
            logger.info(f"SUCCESS at T={T_mid:.4f}, searching higher")
        else:
            # Failed - try lower T
            T_high = T_mid
            logger.info(f"FAILED at T={T_mid:.4f}, searching lower")

    # Final result
    logger.info(f"\n{'='*60}")
    logger.info(f"BINARY SEARCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best T found: {best_T:.4f}")
    logger.info(f"Threshold (T/α): {best_T/(4+epsilon):.4f}")
    logger.info(f"Approximation ratio: 1/{4+epsilon:.2f}")

    if best_allocation is None:
        logger.warning("No feasible allocation found")
        return {agent: set() for agent in instance.agents}

    # Verify final allocation
    logger.info(f"\nFinal allocation verification:")
    threshold = best_T / (4 + epsilon)
    for agent, bundle in best_allocation.items():
        value = sum(instance.agent_item_value(agent, item) for item in bundle)
        logger.info(f"  {agent}: {len(bundle)} items, value {value:.4f} (>= {threshold:.4f})")

    return best_allocation