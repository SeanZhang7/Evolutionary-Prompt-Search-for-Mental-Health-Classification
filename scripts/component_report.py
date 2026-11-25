"""End-of-run component exports, fitness plot and per-component parent similarity."""

import json

if __package__:
    from . import component_prompts as cp
    from .ga_report import CompactReport
    from .prompt_similarity import PromptSimilarityReport
else:
    import component_prompts as cp
    from ga_report import CompactReport
    from prompt_similarity import PromptSimilarityReport


class ComponentReport:
    def __init__(self, output):
        self.output = output
        self.fitness = CompactReport(output)
        self.similarity = PromptSimilarityReport(output)

    def save(self, filename, value):
        (self.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def analyze(self, summary):
        labels = summary["fixed_context"]["labels"]
        result = {"status": "complete", "ga_status": summary["status"], "components": {},
                  "scope": "Each editable component compared with its own participating parent slots; fixed examples excluded. Completed generations only."}
        for name in cp.COMPONENTS:
            population = []
            for item in summary["population"]:
                projected = {**item, "instruction": cp.component_text(name, item["components"][name], labels)}
                if item["generation"]:
                    projected["operations"] = item["component_operations"][name]
                population.append(projected)
            result["components"][name] = self.similarity.analyze({**summary, "population": population})
        return result

    def plot_similarity(self, reports):
        figure = self.similarity.Figure(figsize=(11, 9), constrained_layout=True)
        self.similarity.Canvas(figure)
        axes = figure.subplots(3, 2, sharex=True)
        for row, name in enumerate(cp.COMPONENTS):
            data = reports[name]["generations"]
            x = [r["generation"] for r in data]
            for col, metric in enumerate(("cosine", "bleu")):
                axis = axes[row, col]
                y = [r[f"mean_{metric}"] if r[f"mean_{metric}"] is not None else float("nan") for r in data]
                axis.plot(x, y, marker="o")
                axis.set(title=f"{name}: {'TF-IDF cosine' if metric == 'cosine' else 'BLEU (child → parent)'}",
                         ylim=(0, 1), ylabel="Mean similarity")
                axis.grid(alpha=.25)
                for entry in data:
                    if entry[f"mean_{metric}"] is None:
                        axis.text(entry["generation"], .04, "N/A", ha="center")
                if len(x) == 1:
                    axis.set_xlim(x[0] - .5, x[0] + .5)
                ticks = sorted(set(x[::max(1, (len(x) + 9) // 10)] + [x[-1]]))
                axis.set_xticks(ticks, [f"G{g}" for g in ticks])
        for axis in axes[-1]:
            axis.set_xlabel("Child generation")
        figure.suptitle("Editable component similarity to actual parents (fixed few-shot excluded)")
        figure.savefig(self.output / "prompt_similarity.png", dpi=160)
        figure.clear()

    def __call__(self, summary):
        eligible = [p for p in summary["population"] if p["status"] == "complete"]
        best = min(eligible, key=lambda p: (-p["macro_f1"], p["creation_order"])) if eligible else None
        summary["best"] = best
        summary["best_scope"] = "All complete error-free combinations, including from an unfinished generation"
        summary["plot_scope"] = "Retained-population best and mean, completed generations only"
        summary["fixed_examples_storage"] = "Stored once in fixed_context; every candidate references the same examples_sha256"
        summary["report_errors"] = {}
        self.save("performance.json", summary)
        if best:
            fixed, components = summary["fixed_context"], best["components"]
            self.save("best_task_definition.json", {"label_mode": summary["label_mode"], "text": components["task_definition"],
                                                     "required_output": fixed["required_output"]})
            self.save("best_label_definition.json", {"label_mode": summary["label_mode"], "definitions": components["label_definition"],
                **{k: fixed[k] for k in ("examples", "examples_sha256", "few_shot_intro")}})
            self.save("best_instruction.json", {"description": "Instruction from the best complete combination, not independently scored.",
                                                "prompts": [{"id": best["id"], "instruction": components["instruction"]}]})
            template = cp.render_template(components, fixed).replace("{{INSTRUCTION}}", components["instruction"])
            (self.output / "best_template.txt").write_text(template, encoding="utf-8")
        if summary["generation_metrics"]:
            try:
                self.fitness.plot(summary["generation_metrics"])
            except (Exception, KeyboardInterrupt) as exc:
                summary["report_errors"]["fitness_plot"] = str(exc)
        if summary["status"] != "prepared" and "fixed_context" in summary:
            try:
                similarity = self.analyze(summary)
                self.save("prompt_similarity.json", similarity)
                if summary["completed_generations"]:
                    self.plot_similarity(similarity["components"])
            except (Exception, KeyboardInterrupt) as exc:
                summary["report_errors"]["similarity"] = f"{type(exc).__name__}: {exc}"
                self.save("prompt_similarity.json", {"status": "failed", "error": str(exc)})
        if summary["report_errors"]:
            self.save("performance.json", summary)
            print(f"Component report errors: {summary['report_errors']}", flush=True)
