"""Analyze instruction changes after GA; no model calls or classification data."""

from importlib.metadata import version
import json
import math

TFIDF_PARAMETERS = {
    "lowercase": True, "analyzer": "word", "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b\w+\b", "stop_words": None, "norm": "l2",
    "use_idf": True, "smooth_idf": True, "sublinear_tf": False,
}
BLEU_PARAMETERS = {
    "lowercase": True, "tokenize": "13a", "smooth_method": "exp",
    "max_ngram_order": 4, "effective_order": True,
}
METRICS = ("cosine", "bleu")


def average(values):
    valid = [value for value in values if value is not None]
    return math.fsum(valid) / len(valid) if valid else None


class PromptSimilarityReport:
    def __init__(self, output):
        # Construct before inference so an incomplete environment fails early.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sacrebleu.metrics import BLEU
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise RuntimeError(
                "Prompt similarity requires scikit-learn, sacrebleu and matplotlib. "
                "Install requirements-prompt-search.txt in the prompt-search environment before running GA."
            ) from exc
        self.output = output
        self.Vectorizer = TfidfVectorizer
        self.BLEU = BLEU
        self.Figure = Figure
        self.Canvas = FigureCanvasAgg
        self.versions = {name: version(name) for name in ("scikit-learn", "sacrebleu", "matplotlib")}

    def analyze(self, summary):
        completed = sorted(row["generation"] for row in summary["generation_metrics"]
                           if 0 < row["generation"] <= summary["completed_generations"])
        has_initial = any(row["generation"] == 0 for row in summary["generation_metrics"])
        population = [item for item in summary["population"]
                      if (has_initial and item.get("generation", 0) == 0)
                      or item.get("generation", 0) in completed]
        # Failed generation slots have no individual; an unfinished generation is excluded entirely.
        lookup = {item["id"]: item for item in population}
        if len(lookup) != len(population):
            raise ValueError("Duplicate individual IDs in similarity history")
        texts = list(dict.fromkeys(item["instruction"] for item in population))
        text_indices = {text: index for index, text in enumerate(texts)}
        vectors = None
        vocabulary_size = 0
        if texts:
            vectorizer = self.Vectorizer(**TFIDF_PARAMETERS)
            try:
                vectors = vectorizer.fit_transform(texts)
                vocabulary_size = len(vectorizer.vocabulary_)
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise
        nonzero = vectors.getnnz(axis=1) if vectors is not None else []
        bleu = self.BLEU(**BLEU_PARAMETERS)
        children = []
        for item in population:
            generation = item.get("generation", 0)
            if generation not in completed:
                continue
            parent_ids = item["parent_ids"] if item["operations"]["crossover"] else item["parent_ids"][:1]
            if not parent_ids:
                raise ValueError(f"Missing parent IDs for {item['id']}")
            pairs = []
            child_index = text_indices[item["instruction"]]
            for parent_id in parent_ids:
                parent = lookup[parent_id]
                if parent.get("generation", 0) >= generation:
                    raise ValueError(f"Parent {parent_id} must predate {item['id']}")
                parent_index = text_indices[parent["instruction"]]
                cosine = None
                if vectors is not None and nonzero[child_index] and nonzero[parent_index]:
                    cosine = float(vectors[child_index].multiply(vectors[parent_index]).sum())
                    cosine = min(1.0, max(0.0, cosine))
                score = bleu.sentence_score(item["instruction"], [parent["instruction"]]).score / 100
                pairs.append({"parent_id": parent_id, "cosine": cosine, "bleu": score})
            children.append({
                "child_id": item["id"], "generation": generation,
                "parent_ids": item["parent_ids"], "compared_parent_ids": parent_ids,
                "operations": item["operations"], "status": item["status"],
                "copied": not any(item["operations"].values()),
                "skipped_duplicate": item["status"] == "skipped_duplicate",
                "duplicate_of": item.get("duplicate_of"), "pairs": pairs,
                **{f"mean_{metric}": average(pair[metric] for pair in pairs) for metric in METRICS},
                "valid_pair_counts": {metric: sum(pair[metric] is not None for pair in pairs) for metric in METRICS},
            })
        generations = []
        for generation in completed:
            items = [item for item in children if item["generation"] == generation]
            generations.append({
                "generation": generation, "child_count": len(items),
                "copied_count": sum(item["copied"] for item in items),
                "duplicate_count": sum(item["skipped_duplicate"] for item in items),
                "invalid_generation_slots": sum(slot["generation"] == generation
                                                for slot in summary.get("skipped_generation_slots", [])),
                **{f"mean_{metric}": average(item[f"mean_{metric}"] for item in items) for metric in METRICS},
                "valid_child_counts": {metric: sum(item[f"mean_{metric}"] is not None for item in items)
                                       for metric in METRICS},
                "missing_child_counts": {metric: sum(item[f"mean_{metric}"] is None for item in items)
                                         for metric in METRICS},
            })
        return {
            "status": "complete", "ga_status": summary["status"], "error": None,
            "completed_generations": completed, "label_mode": summary.get("label_mode"),
            "scope": "Initial instructions and valid children in completed generations only; includes copies and skipped duplicates.",
            "aggregation": "Mean over participating parent slots per child, then mean over children per generation; missing values excluded per metric.",
            "interpretation": "Lexical overlap, not classification quality or semantic equivalence. BLEU is directional; TF-IDF is fitted within this run.",
            "parameters": {"tfidf": TFIDF_PARAMETERS, "bleu": BLEU_PARAMETERS,
                           "bleu_scale": "sentence BLEU / 100", "bleu_direction": "child candidate, parent reference",
                           "tfidf_fit": "exact unique instruction texts, initial population plus valid children of completed generations",
                           "empty_vector_policy": "cosine is null when either vector is zero or vocabulary is empty"},
            "versions": self.versions, "unique_text_count": len(texts), "vocabulary_size": vocabulary_size,
            "prompts": {item["id"]: {"instruction": item["instruction"], "generation": item.get("generation", 0)}
                        for item in population},
            "children": children, "generations": generations,
        }

    def plot(self, generations):
        figure = self.Figure(figsize=(7.2, 6.0), constrained_layout=True)
        self.Canvas(figure)
        axes = figure.subplots(2, 1, sharex=True)
        x = [row["generation"] for row in generations]
        for axis, metric, title in zip(axes, METRICS, ("TF-IDF cosine", "BLEU (child → parent)")):
            y = [row[f"mean_{metric}"] if row[f"mean_{metric}"] is not None else float("nan")
                 for row in generations]
            axis.plot(x, y, marker="o", linewidth=2)
            axis.set(title=title, ylabel="Mean similarity", ylim=(0, 1))
            axis.grid(alpha=0.25)
            for row in generations:
                if row[f"mean_{metric}"] is None:
                    axis.text(row["generation"], 0.04, "N/A", transform=axis.get_xaxis_transform(),
                              ha="center", fontsize=8)
            if len(x) == 1:
                axis.set_xlim(x[0] - 0.5, x[0] + 0.5)
        step = max(1, (len(x) + 9) // 10)
        ticks = sorted(set(x[::step] + [x[-1]]))
        axes[-1].set_xticks(ticks, [f"G{value}" for value in ticks])
        axes[-1].set_xlabel("Child generation (compared with actual parents)")
        figure.suptitle("Prompt similarity to parents")
        temporary = self.output / ".prompt_similarity.png.tmp"
        try:
            figure.savefig(temporary, format="png", dpi=160)
            temporary.replace(self.output / "prompt_similarity.png")
        finally:
            temporary.unlink(missing_ok=True)
            figure.clear()

    def __call__(self, summary):
        # The main GA report is already saved; analysis failures cannot remove it.
        report = {"status": "failed", "ga_status": summary["status"], "error": None}
        try:
            report = self.analyze(summary)
            if report["generations"]:
                self.plot(report["generations"])
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            temporary = self.output / ".prompt_similarity.json.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(self.output / "prompt_similarity.json")
        except OSError as exc:
            report.update(status="failed", error=f"Cannot save similarity report: {exc}")
        print(f"Similarity analysis: {report['status']}" + (f" — {report['error']}" if report["error"] else ""), flush=True)
        return report
