"""
Compare runtime of qp_max_min_allocation before and after lru_cache optimization.

Uses the existing CSV results as the "before" baseline, runs the optimized
code on the same inputs for the "after" data, then scales up to find
the largest input computable in ~60 seconds.

Programmer: Rotem Melamed
Date: 2026-03-10
"""

import pandas as pd
import numpy as np
import random
import time
import logging
import matplotlib.pyplot as plt
from fairpyx.instances import Instance
from fairpyx.algorithms.qp_local_search import qp_max_min_allocation

max_value = 1000
TIME_LIMIT = 60

logging.getLogger("QP_Local_Search").setLevel(logging.WARNING)


def random_binary_instance(num_of_players: int, num_of_gifts: int, max_value: int) -> Instance:
    """Creates a random binary instance (same logic as compare_qp_local_search.py)."""
    agents = [f"P{i+1}" for i in range(num_of_players)]
    items  = [f"c{j+1}" for j in range(num_of_gifts)]
    base_values = {item: random.randint(1, max_value) for item in items}
    valuations = {a: {} for a in agents}

    for a in agents:
        k = random.randint(1, num_of_gifts)
        chosen = random.sample(items, k=k)
        for item in items:
            valuations[a][item] = base_values[item] if item in chosen else 0

    for item in items:
        if not any(valuations[a][item] > 0 for a in agents):
            a = random.choice(agents)
            valuations[a][item] = base_values[item]

    agent_caps = {a: num_of_gifts for a in agents}
    item_caps  = {i: 1 for i in items}
    return Instance(valuations=valuations, agent_capacities=agent_caps, item_capacities=item_caps)


def run_optimized_matching_experiments():
    """
    Run the optimized algorithm on the same inputs as the original experiment
    and collect runtime data.
    """
    # Same parameter grid as the original experiment
    players_list = [3, 4, 5, 6]
    gifts_list = [6, 8, 10, 12, 14]
    seeds = list(range(5))

    results = []
    for num_players in players_list:
        for num_gifts in gifts_list:
            for seed in seeds:
                np.random.seed(seed)
                random.seed(seed)
                instance = random_binary_instance(num_players, num_gifts, max_value)

                start = time.time()
                qp_max_min_allocation(instance)
                elapsed = time.time() - start

                results.append({
                    "num_of_players": num_players,
                    "num_of_gifts": num_gifts,
                    "random_seed": seed,
                    "runtime": elapsed,
                })
                print(f"  players={num_players}, gifts={num_gifts}, seed={seed} -> {elapsed:.4f}s")

    return pd.DataFrame(results)


def plot_runtime_comparison(before_df, after_df):
    """
    Plot runtime comparison (before vs after optimization) as a function
    of input size (num_players * num_gifts).
    """
    # Compute input_size and mean runtime for each input_size
    before_df = before_df.copy()
    before_df["input_size"] = before_df["num_of_players"] * before_df["num_of_gifts"]
    before_mean = before_df.groupby("input_size")["runtime"].mean().sort_index()

    after_df = after_df.copy()
    after_df["input_size"] = after_df["num_of_players"] * after_df["num_of_gifts"]
    after_mean = after_df.groupby("input_size")["runtime"].mean().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(before_mean.index, before_mean.values, 'o-', label="Before optimization (no cache)", color="tab:orange")
    plt.plot(after_mean.index, after_mean.values, 'o-', label="After optimization (lru_cache)", color="tab:blue")
    plt.xlabel("Input size (num_players × num_gifts)")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("qp_max_min_allocation: Runtime Before vs After lru_cache Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/results/qp_optimized_runtime_comparison.png", dpi=150)
    plt.close()
    print("Saved: results/qp_optimized_runtime_comparison.png")


def find_max_input_size():
    """
    Scale up input size to find the largest instance computable in ~60 seconds.
    Uses a fixed ratio of gifts = 2 * players to scale uniformly.
    """
    print("\n=== Finding max input size for ~60 seconds ===")
    results = []

    # Start at 100 players (gifts = 2*players), multiply by 3 each step
    num_players = 100
    test_configs = []
    while num_players <= 300000:
        test_configs.append((num_players, num_players * 2))
        num_players *= 3

    total_start = time.time()
    for i, (num_players, num_gifts) in enumerate(test_configs):
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        instance = random_binary_instance(num_players, num_gifts, max_value)

        total_elapsed = time.time() - total_start
        print(f"  [{i+1}/{len(test_configs)}] [total: {total_elapsed:.0f}s] "
              f"players={num_players}, gifts={num_gifts} (size={num_players*num_gifts})...", end=" ", flush=True)
        start = time.time()
        qp_max_min_allocation(instance)
        elapsed = time.time() - start
        print(f"{elapsed:.2f}s")

        results.append({
            "num_of_players": num_players,
            "num_of_gifts": num_gifts,
            "input_size": num_players * num_gifts,
            "runtime": elapsed,
        })

        if elapsed > TIME_LIMIT:
            print(f"  Exceeded {TIME_LIMIT}s limit. Stopping.")
            break

    df = pd.DataFrame(results)

    # Plot scaling curve
    plt.figure(figsize=(10, 6))
    plt.plot(df["input_size"], df["runtime"], 'o-', color="tab:blue", label="Optimized (lru_cache)")
    plt.axhline(y=TIME_LIMIT, color='red', linestyle='--', label=f"{TIME_LIMIT}s limit")
    plt.xlabel("Input size (num_players × num_gifts)")
    plt.ylabel("Runtime (seconds)")
    plt.title("qp_max_min_allocation (optimized): Scaling to ~60s Limit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/results/qp_optimized_max_input_size.png", dpi=150)
    plt.close()
    print("Saved: results/qp_optimized_max_input_size.png")

    # Report the run that crossed ~60s
    over_limit = df[df["runtime"] > TIME_LIMIT]
    if not over_limit.empty:
        crossed = over_limit.iloc[0]
        print(f"\n  Biggest input computed in ~{TIME_LIMIT}s: "
              f"players={int(crossed['num_of_players'])}, gifts={int(crossed['num_of_gifts'])} "
              f"(size={int(crossed['input_size'])}), runtime={crossed['runtime']:.2f}s")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # --- Part 1: Runtime comparison before vs after optimization ---
    print("=== Part 1: Runtime comparison (before vs after lru_cache) ===\n")

    # Load "before" data from existing CSV
    before_df = pd.read_csv("experiments/results/qpVSsanta_experiment.csv")
    before_df = before_df[before_df["algorithm"] == "qp_max_min_allocation"]
    print(f"Loaded {len(before_df)} 'before' results from existing CSV.\n")

    # Run optimized version on same inputs
    print("Running optimized algorithm on same inputs...")
    after_df = run_optimized_matching_experiments()

    # Plot comparison
    plot_runtime_comparison(before_df, after_df)

    # --- Part 2: Find max input size in ~60s ---
    find_max_input_size()
