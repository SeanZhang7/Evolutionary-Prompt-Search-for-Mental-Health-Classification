"""Compact end-of-run reports for two maximized task fitness values."""

import json

if __package__:
    from .multiobjective import TASKS, pareto_front
else:
    from multiobjective import TASKS, pareto_front


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def representative(population):
    return min(population, key=lambda p: (-sum(p["fitness"].values()) / 2,
                                          -min(p["fitness"].values()), p["creation_order"])) if population else None


def plot_fitness(output, history):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 7), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    generations = [row["generation"] for row in history]
    for axis, task in zip(axes, TASKS):
        best = [row["tasks"][task]["best_macro_f1"] for row in history]
        mean = [row["tasks"][task]["mean_macro_f1"] for row in history]
        axis.plot(generations, best, marker="o", label="Best")
        axis.plot(generations, mean, marker="s", linestyle="--", label="Mean")
        axis.set(title=task.capitalize(), ylabel="Macro-F1", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.25)
        axis.legend()
        if len(generations) == 1:
            axis.set_xlim(-0.5, 0.5)
    step = max(1, (len(generations) + 9) // 10)
    ticks = sorted(set(generations[::step] + [generations[-1]]))
    axes[-1].set_xticks(ticks, [f"G{g}" for g in ticks])
    axes[-1].set_xlabel("Generation (G0 = initial population)")
    figure.suptitle("Best and mean fitness of retained population")
    figure.savefig(output / "best_fitness.png", dpi=160)
    figure.clear()


def plot_pareto(output, population, front, best):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(6.5, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.scatter([p["fitness"][TASKS[0]] for p in population],
                 [p["fitness"][TASKS[1]] for p in population], alpha=0.35, label="Fully evaluated candidates")
    axis.scatter([p["fitness"][TASKS[0]] for p in front],
                 [p["fitness"][TASKS[1]] for p in front], marker="D", label="Observed Pareto front")
    axis.scatter([best["fitness"][TASKS[0]]], [best["fitness"][TASKS[1]]],
                 marker="*", s=180, color="black", label="Representative (highest mean F1)")
    axis.set(xlabel="DepSign binary macro-F1", ylabel="Dreaddit macro-F1",
             xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), title="Joint search on development data")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "pareto_front.png", dpi=160)
    figure.clear()


def write_report(output, summary):
    eligible = [p for p in summary["population"] if p["status"] == "complete" and set(p["fitness"]) == set(TASKS)]
    front = pareto_front(eligible)
    best = representative(eligible)
    summary["best"] = ({**best, "mean_task_macro_f1": sum(best["fitness"].values()) / 2} if best else None)
    summary["representative_policy"] = "Highest arithmetic mean of two task F1 values; ties: higher minimum, then creation order. Reporting only, never used in selection."
    summary["pareto_front_ids"] = [p["id"] for p in front]
    summary["pareto_scope"] = "All fully evaluated candidates, including completed candidates from an unfinished generation"
    summary["plot_scope"] = "Completed generations only; each task's best may belong to a different instruction"
    summary["plot_errors"] = {}
    # Save measured results before optional plots, including on interrupted runs.
    write_json(output / "performance.json", summary)
    write_json(output / "pareto_front.json", {"tasks": list(TASKS), "direction": "maximize",
                                              "ga_status": summary["status"], "scope": summary["pareto_scope"],
                                              "prompts": front})
    if best:
        (output / "best_prompt.txt").write_text(best["instruction"] + "\n", encoding="utf-8")
    plots = []
    if summary["generation_metrics"]:
        plots.append(("best_fitness.png", lambda: plot_fitness(output, summary["generation_metrics"])))
    if eligible:
        plots.append(("pareto_front.png", lambda: plot_pareto(output, eligible, front, best)))
    for name, plot in plots:
        try:
            plot()
        except (Exception, KeyboardInterrupt) as exc:
            summary["plot_errors"][name] = f"{type(exc).__name__}: {exc}"
            print(f"Could not plot {name}: {exc}", flush=True)
    if summary["plot_errors"]:
        write_json(output / "performance.json", summary)
