"""Small GA report: best instruction, performance metadata, generation best-fitness plot."""

import json


class CompactReport:
    def __init__(self, output):
        # Import before starting inference, so missing plotting dependencies fail early.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.output = output
        self.Figure = Figure
        self.Canvas = FigureCanvasAgg

    def __call__(self, summary):
        # Scores with transport errors are not eligible as a best measured candidate.
        eligible = [item for item in summary["population"] if item["status"] == "complete"]
        best = min(eligible, key=lambda item: (-item["macro_f1"], item["creation_order"])) if eligible else None
        best_metrics = None
        if best is not None:
            row = next(row for row in summary["results"] if row["prompt_id"] == best["id"])
            best_metrics = {"id": best["id"], "source_id": best.get("source_id"),
                            "generation": best.get("generation", 0),
                            "macro_f1": best["macro_f1"], "n": row["n"],
                            "invalid_output_rate": row["invalid_output_rate"],
                            "request_error_count": row["request_error_count"]}
            self.write("best_prompt.txt", best["instruction"] + "\n")
        history = summary.get("generation_metrics", [])
        report = {
            "status": summary["status"], "error": summary["error"], "stage": summary["stage"],
            "purpose": summary["purpose"], "seed": summary["seed"],
            "label_mode": summary.get("label_mode", "three_class"),
            "labels": summary.get("labels"), "label_mapping": summary.get("label_mapping"),
            "best": best_metrics,
            "best_scope": "completed_error_free_candidates_in_this_run",
            "prompt_content": "instruction block inserted into the fixed task template",
            "data": summary.get("sources", {}).get("data_snapshot.tsv"),
            "task_template": summary.get("sources", {}).get("task_template.txt"),
            "context_tokens": summary.get("context_tokens"),
            "operator_prompt": summary.get("operator_prompt"),
            "operator_feedback": summary.get("operator_feedback", "none"),
            "feedback_preparation": summary.get("feedback_preparation"),
            "generation_parameters": summary.get("generation_parameters"),
            "classification_parameters": summary.get("classification_parameters"),
            "candidate_evaluations": summary["candidate_evaluations"],
            "population_size": summary.get("population_size"),
            "children_per_request": summary.get("children_per_request", 1),
            "generation_groups_per_generation": summary.get("generation_groups_per_generation"),
            "offspring_per_generation": summary.get("offspring_per_generation"),
            "generation_proposal_slots": summary.get("generation_proposal_slots"),
            "invalid_generation_groups": summary.get("invalid_generation_groups", 0),
            "generation_requests": summary["generation_requests"],
            "last_generation_issue": summary.get("last_generation_issue"),
            "requested_generations": summary.get("requested_generations"),
            "completed_generations": summary.get("completed_generations", 0),
            "stopped_early": summary.get("stopped_early", False),
            "stop_reason": summary.get("stop_reason"),
            "early_stopping_patience": summary["rules"].get("early_stopping_patience"),
            "parent_selection": summary["rules"].get("parent_selection"),
            "tournament_size": summary["rules"].get("tournament_size"),
            "tournament_tie_break": summary["rules"].get("tournament_tie_break"),
            "unchanged_mean_generations": summary.get("unchanged_mean_generations", 0),
            "crossover_probability": summary["rules"].get("crossover_probability"),
            "mutation_probability": summary["rules"].get("mutation_probability"),
            "deduplication": summary["rules"].get("deduplication"),
            "input_prompt_count": summary.get("input_prompt_count"),
            "initial_duplicates_removed": summary.get("initial_duplicates_removed", 0),
            "duplicate_offspring_skipped": summary.get("duplicate_offspring_skipped", 0),
            "invalid_offspring_skipped": summary.get("invalid_offspring_skipped", 0),
            "skipped_generation_slots": summary.get("skipped_generation_slots", []),
            "exhausted_slot_action": summary["rules"].get("exhausted_slot_action"),
            "max_crossover_tokens": summary["rules"].get("max_crossover_tokens"),
            "max_instruction_tokens": summary["rules"].get("max_instruction_tokens"),
            "deduplication_stage": summary.get("deduplication_stage"),
            "deduplication_scope": summary.get("deduplication_scope"),
            "copied_offspring": summary.get("copied_offspring", 0),
            "candidate_evaluation_budget": summary.get("candidate_evaluation_budget"),
            "budget_interpretation": summary.get("budget_interpretation"),
            "planned_classification_requests": summary.get("planned_classification_requests"),
            "completed_classification_requests": summary.get("logged_classification_requests", 0),
            "evaluation_inputs": summary.get("evaluation_inputs"),
            "generation_metrics": history,
            "plot_metric": "best_macro_f1",
            "best_fitness_definition": "Maximum individual macro-F1 in the retained population of each completed generation. G0 is initialization.",
            "mean_f1_definition": "Arithmetic mean of individual macro-F1 in the retained population after configured deduplication and selection. G0 is initialization.",
        }
        self.write("performance.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if history:
            self.plot(history)

    def write(self, name, contents):
        temporary = self.output / ("." + name + ".tmp")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(self.output / name)

    def plot(self, history):
        figure = self.Figure(figsize=(6.4, 4.0), constrained_layout=True)
        self.Canvas(figure)
        axis = figure.subplots()
        generations = [row["generation"] for row in history]
        best_scores = [row["best_macro_f1"] for row in history]
        axis.plot(generations, best_scores, marker="o", linewidth=2)
        label_step = max(1, (len(generations) + 9) // 10)
        labeled = sorted(set(range(0, len(generations), label_step)) | {len(generations) - 1})
        for index in labeled:
            generation, best = generations[index], best_scores[index]
            axis.annotate(f"{best:.4f}", (generation, best), xytext=(0, 9),
                          textcoords="offset points", ha="center")
        axis.set(xlabel="Generation (G0 = initial population)", ylabel="Best fitness (macro-F1)",
                 title="Best fitness in each generation", ylim=(-0.05, 1.08))
        ticks = [generations[index] for index in labeled]
        axis.set_xticks(ticks, [f"G{value}" for value in ticks])
        if len(generations) == 1:
            axis.set_xlim(generations[0] - 0.5, generations[0] + 0.5)
        axis.grid(alpha=0.25)
        temporary = self.output / ".best_fitness.png.tmp"
        figure.savefig(temporary, format="png", dpi=160)
        temporary.replace(self.output / "best_fitness.png")
        figure.clear()
