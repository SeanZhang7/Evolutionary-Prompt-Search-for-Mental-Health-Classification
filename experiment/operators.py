"""Proposal operators for the DepSign experiment.

The textual DE procedure follows Guo et al. (ICLR 2024), Figure 2 and Algorithm 3:
https://openreview.net/forum?id=ZG3RaNIsO8
https://github.com/beeevita/EvoPrompt/blob/main/data/template_de.py
The generation snapshot supplies the best prompt for every DE slot. Upstream code
can refresh the best during the generation. We omit their demonstrations and
use the same English instruction / JSON contract for all three methods.
"""

import json
import math
import random
from collections.abc import Sequence


METHODS = ("de", "ga", "random")
CONTRACT = (
    "Write one concise English instruction for classifying a social media post. "
    "The classifier receives the task, label definitions, post and required output "
    "separately. Write actionable guidance that can be used with those definitions. "
    "Do not include examples, dataset or label names, placeholders or an answer to a post. "
    'Return exactly one JSON object with one string field: {"instruction":"..."}.'
)
EDIT = {
    "de": (
        "Compare donor_a and donor_b and identify where their decision strategies or "
        "wording differ. Mutate only those differences into alternative instructions. "
        "Select promising changes and apply them to best to form an intermediate draft. "
        "Cross this draft with target to produce one trial instruction. "
        "Return the final trial only."
    ),
    "ga": (
        "Cross parent_a with parent_b by recombining their useful guidance. "
        "Mutate the resulting instruction to give the classifier a distinct way "
        "of making its decision. Return the final child only."
    ),
    "random": (
        "Independently propose a decision procedure for the classifier described "
        "above. Return one instruction."
    ),
}


def _check_inputs(method, population, scores, target_index):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    minimum = {"de": 3, "ga": 2, "random": 1}[method]
    if len(population) < minimum or len(scores) != len(population):
        raise ValueError(f"{method} needs at least {minimum} scored individuals")
    if not 0 <= target_index < len(population):
        raise ValueError("Target index outside generation snapshot")
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise ValueError("Expected finite macro-F1 scores between 0 and 1")


def _roulette_pool(scores, rng):
    weights = scores if any(scores) else None
    return rng.choices(range(len(scores)), weights=weights, k=len(scores))


def build_proposal(
    method: str,
    population: Sequence[str],
    scores: Sequence[float],
    target_index: int,
    rng: random.Random,
    *,
    wheel_pool=None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return chat messages and role-to-slot indices for a single proposal."""
    _check_inputs(method, population, scores, target_index)
    indices = range(len(population))
    if method == "de":
        donor_a, donor_b = rng.sample([i for i in indices if i != target_index], 2)
        roles = {
            "target": target_index,
            "donor_a": donor_a,
            "donor_b": donor_b,
            "best": max(indices, key=scores.__getitem__),
        }
    elif method == "ga":
        if wheel_pool is None:
            wheel_pool = _roulette_pool(scores, rng)
        parent_a, parent_b = rng.sample(wheel_pool, 2)
        roles = {"parent_a": parent_a, "parent_b": parent_b}
    else:
        roles = {}
    instruction = EDIT[method]
    if roles:
        sources = {role: population[index] for role, index in roles.items()}
        instruction += "\nSource instructions (JSON):\n" + json.dumps(sources, ensure_ascii=False)
    return [
        {"role": "system", "content": CONTRACT},
        {"role": "user", "content": instruction},
    ], roles


def build_generation_proposals(method, population, scores, rng):
    """Draw a generation before any of its trials are evaluated."""
    _check_inputs(method, population, scores, 0)
    pool = _roulette_pool(scores, rng) if method == "ga" else None
    return [
        build_proposal(method, population, scores, slot, rng, wheel_pool=pool)
        for slot in range(len(population))
    ]


def parse_proposal(content: str) -> str:
    """Require a single instruction field. Token length is checked by the evaluator."""
    if not isinstance(content, str):
        raise ValueError("No textual proposal")
    # Preserve duplicate keys. Objects become tuples of pairs and arrays remain lists.
    obj = json.loads(content, object_pairs_hook=tuple)
    if not isinstance(obj, tuple) or len(obj) != 1 or obj[0][0] != "instruction":
        raise ValueError("Expected a JSON object with exactly one instruction field")
    value = obj[0][1]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty instruction string")
    return value.strip()
