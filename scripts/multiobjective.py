"""NSGA-II rank/crowding selection for two maximized task macro-F1 values.

Reference: https://pymoo.org/algorithms/moo/nsga2.html
No scalarized fitness or text similarity is used in selection.
"""

import math
import random

TASKS = ("depsign", "dreaddit")


def dominates(a, b):
    return all(a[t] >= b[t] for t in TASKS) and any(a[t] > b[t] for t in TASKS)


def rank_and_crowding(population):
    """Return fronts of indices and per-index ranks/distances without mutating individuals."""
    if len({p["id"] for p in population}) != len(population):
        raise ValueError("Duplicate individual IDs")
    for item in population:
        if (set(item["fitness"]) != set(TASKS)
                or any(not math.isfinite(item["fitness"][t]) or not 0 <= item["fitness"][t] <= 1
                       for t in TASKS)):
            raise ValueError("Require two finite task fitness values in [0, 1]")
    dominated = [[] for _ in population]
    counts = [0] * len(population)
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            a, b = population[i]["fitness"], population[j]["fitness"]
            if dominates(a, b):
                dominated[i].append(j)
                counts[j] += 1
            elif dominates(b, a):
                dominated[j].append(i)
                counts[i] += 1
    fronts, rank = [], {}
    front = [i for i, count in enumerate(counts) if count == 0]
    while front:
        front.sort(key=lambda i: population[i]["creation_order"])
        for i in front:
            rank[i] = len(fronts)
        fronts.append(front)
        following = []
        for i in front:
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        front = following
    distance = {i: 0.0 for i in range(len(population))}
    for front in fronts:
        for task in TASKS:
            ordered = sorted(front, key=lambda i: (population[i]["fitness"][task],
                                                  population[i]["creation_order"]))
            low, high = (population[ordered[k]]["fitness"][task] for k in (0, -1))
            if high == low:
                continue
            distance[ordered[0]] = distance[ordered[-1]] = math.inf
            for k in range(1, len(ordered) - 1):
                distance[ordered[k]] += (population[ordered[k + 1]]["fitness"][task]
                                         - population[ordered[k - 1]]["fitness"][task]) / (high - low)
    return fronts, rank, distance


def select_parents(population, seed):
    if len(population) < 2:
        raise ValueError("Require at least two parents")
    _, ranks, distances = rank_and_crowding(population)
    rng = random.Random(seed)
    pairs, tournaments = [], []
    for _ in population:
        pair, groups = [], []
        for _ in range(2):
            a, b = rng.sample(range(len(population)), 2)
            groups.append([population[a]["id"], population[b]["id"]])
            ka, kb = (ranks[a], -distances[a]), (ranks[b], -distances[b])
            winner = rng.choice((a, b)) if ka == kb else a if ka < kb else b
            pair.append(population[winner]["id"])
        pairs.append(pair)
        tournaments.append(groups)
    return {"mode": "nsga2_rank_crowding_tournament", "seed": seed,
            "parent_ids": pairs, "tournaments": tournaments}


def select_survivors(population, count):
    # Input order determines which exact instruction copy is kept.
    seen, unique = set(), []
    for p in population:
        if p["instruction"].strip() not in seen:
            unique.append(p)
            seen.add(p["instruction"].strip())
    if type(count) is not int or not 1 <= count <= len(unique):
        raise ValueError("Not enough unique candidates for survivor count")
    fronts, _, distances = rank_and_crowding(unique)
    selected = []
    for front in fronts:
        remaining = count - len(selected)
        if len(front) > remaining:
            front = sorted(front, key=lambda i: (-distances[i], unique[i]["creation_order"]))[:remaining]
        selected.extend(unique[i] for i in front)
        if len(selected) == count:
            break
    return selected


def pareto_front(population):
    fronts, _, _ = rank_and_crowding(population)
    return [population[i] for i in fronts[0]] if fronts else []
