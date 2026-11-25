"""Frozen DepSign classification and paired development splits for search experiments."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import depsign_evaluator as ev
from scripts.run_depsign_pipeline_smoke import default_prompts, load_prompts, model_identity


PROMPTS = ROOT / "prompts/depsign"
DATA = ROOT / "data/processed/depsign"
# Nine IDs in the original single-instruction smoke run, 20261020T021800892846Z.
HISTORICAL_SMOKE_IDS = {
    f"dev_pid_{i}" for i in (4223, 1684, 3650, 4132, 1878, 4166, 4478, 2057, 2400)
}


def stratified(rows, size, seed):
    """Preserve raw class proportions while representing all three raw labels."""
    labels = list(ev.LABEL_MAP)
    groups = {label: [row for row in rows if row["Label"] == label] for label in labels}
    total = len(rows)
    if not len(labels) <= size <= total or any(not group for group in groups.values()):
        raise ValueError(f"Sample size must be between {len(labels)} and {total}, with all raw classes available")
    if size == total:
        return list(rows)
    # Starting at one per class prevents the only severe record disappearing in small pilots.
    # Repeatedly fill the largest deficit relative to the target proportional allocation.
    quotas = dict.fromkeys(labels, 1)
    for _ in range(size - len(labels)):
        available = [label for label in labels if quotas[label] < len(groups[label])]
        label = max(available, key=lambda key: size * len(groups[key]) - total * quotas[key])
        quotas[label] += 1
    rng = random.Random(seed)
    selected = [row for label in labels for row in rng.sample(groups[label], quotas[label])]
    rng.shuffle(selected)
    return selected


def make_splits(mode, template, search_size, confirmation_size, split_seed):
    clean = ev.load_data(ROOT / ev.CLEAN_DEV)
    dev = ev.load_data(DATA / "dev_100.tsv")
    expected = json.loads((DATA / "dev_100.manifest.json").read_text())
    if ev.digest(dev["data"]) != expected["sample_sha256"] or ev.digest(clean["data"]) != expected["source_sha256"]:
        raise ValueError("Local data differ from the dev_100 sampling manifest")
    clean_by_id = {row["PID"]: row for row in clean["rows"]}
    if any(clean_by_id.get(row["PID"]) != row for row in dev["rows"]):
        raise ValueError("A dev_100 example differs from its cleaned source")
    examples = json.loads((PROMPTS / "components" / mode / "label_definition.json").read_text())["examples"]
    mapping = ev.task_spec(mode)["label_mapping"]
    for example in examples:
        source = clean_by_id[example["id"]]
        quoted = json.dumps(example["text"], ensure_ascii=False)
        if (source["Text data"] != example["text"] or mapping[source["Label"]] != example["label"]
                or f'{quoted}\nLabel: {example["label"]}' not in template):
            raise ValueError("The fixed few-shot example differs from the supplied template or source")
    if template.count("\nPost (JSON string)\n") != len(examples) + 1:
        raise ValueError("Untracked few-shot example in the template")
    demo_ids = {example["id"] for example in examples}
    demo_texts = {example["text"].strip() for example in examples}
    if any(row["PID"] in demo_ids or row["Text data"].strip() in demo_texts for row in dev["rows"]):
        raise ValueError("Search data overlaps a fixed demonstration")
    excluded_ids = demo_ids | HISTORICAL_SMOKE_IDS | {row["PID"] for row in dev["rows"]}
    excluded_texts = {clean_by_id[pid]["Text data"].strip() for pid in excluded_ids}
    eligible = [row for row in clean["rows"] if row["PID"] not in excluded_ids
                and row["Text data"].strip() not in excluded_texts]
    splits = {
        "search": stratified(dev["rows"], search_size, split_seed),
        "confirmation": stratified(eligible, confirmation_size, split_seed + 1),
    }
    manifest = {
        "clean_dev_sha256": ev.digest(clean["data"]),
        "dev_100_sha256": ev.digest(dev["data"]),
        "split_seed": split_seed,
        "confirmation_seed": split_seed + 1,
        "sampling": "Proportional raw-label counts, minimum one per raw label, without replacement; original order for full dev_100.",
        "confirmation_excluded": {"all_dev_100": len(dev["rows"]), "few_shot_ids": sorted(demo_ids),
                                  "earlier_smoke_ids": sorted(HISTORICAL_SMOKE_IDS), "eligible": len(eligible)},
        "scope": "Development confirmation, disjoint from all dev_100 and the few-shot/known smoke posts by PID and stripped text. Earlier full-dev use means it is not a previously unseen benchmark test.",
    }
    for split, rows in splits.items():
        manifest[split] = {"n": len(rows), "ids": [row["PID"] for row in rows],
                           "class_counts": dict(Counter(mapping[row["Label"]] for row in rows))}
    return splits, manifest


class ClassificationError(RuntimeError):
    def __init__(self, message, requests, usage):
        super().__init__(message)
        self.classification_requests = requests
        self.usage = dict(usage)


class Evaluation:
    """Preflight and classify only on explicit calls; construction uses local files only."""

    def __init__(self, label_mode="three_class", search_size=100, confirmation_size=100,
                 split_seed=20261125, workers=1):
        ev.task_spec(label_mode)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.label_mode = label_mode
        path = PROMPTS / ("depsign_task_Bi.txt" if label_mode == "binary" else "depsign_task_v2.txt")
        self.template = path.read_text(encoding="utf-8")
        self.splits, self.manifest = make_splits(label_mode, self.template, search_size, confirmation_size, split_seed)
        self.tokenizer = ev.load_tokenizer()
        prompts_path = default_prompts(label_mode)
        initial = load_prompts(prompts_path.read_bytes())
        self.initial_instructions = [ev.validate_instruction(p["instruction"], self.tokenizer)[0] for p in initial]
        self.manifest.update(label_mode=label_mode, template=str(path.relative_to(ROOT)),
                             template_sha256=ev.digest(path.read_bytes()), initial_prompts=str(prompts_path.relative_to(ROOT)),
                             initial_prompts_sha256=ev.digest(prompts_path.read_bytes()), model=ev.MODEL,
                             context_tokens=ev.CONTEXT_TOKENS, text_token_limit=ev.TEXT_TOKENS,
                             classification_parameters=ev.PARAMETERS, workers=workers)

    def preflight(self):
        models = json.loads(ev.http_call("/v1/models"))
        identity = model_identity({"server_models": models, "model": ev.MODEL})
        if identity["max_model_len"] is not None and identity["max_model_len"] < ev.CONTEXT_TOKENS:
            raise ValueError("Served context is smaller than PROMPT_SEARCH_CONTEXT_TOKENS")
        return {"identity": identity, "version": json.loads(ev.http_call("/version"))}

    def evaluate(self, instruction, split="search"):
        prepared = ev.prepare_candidate(instruction, self.splits[split], self.tokenizer,
                                        task_template=self.template, label_mode=self.label_mode)["requests"]
        results = []
        usage = Counter()
        request_count = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Bound in-flight requests so an endpoint failure cannot flood the rest of a split.
            for start in range(0, len(prepared), self.workers):
                batch = list(executor.map(ev.classify, prepared[start:start + self.workers]))
                request_count += len(batch)
                for item in batch:
                    counts = item.get("usage") or {}
                    usage.update({k: counts[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in counts})
                for item in batch:
                    if item["request_error"]:
                        raise ClassificationError(f"Classification failed for {item['PID']}: {item['request_error']}",
                                                  request_count, usage)
                    if item["reasoning"] or item["finish_reason"] not in ("stop", "length"):
                        raise ClassificationError(f"Unexpected classification response for {item['PID']} "
                                                  f"(finish={item['finish_reason']}, reasoning={bool(item['reasoning'])})",
                                                  request_count, usage)
                    results.append({key: item[key] for key in ("PID", "gold", "prediction", "finish_reason")})
        metrics = ev.calculate_metrics(results, self.label_mode)
        metrics.update(predictions=results, usage=dict(usage), classification_requests=len(results),
                       truncated_count=sum(item["truncated"] for item in prepared))
        return metrics
