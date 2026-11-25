"""Run the full DepSign three-component search with a selected survivor policy."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import component_ga, run_ga


POLICIES = {
    "elite": ("mu_plus_lambda_top_mu", "creation_order_parents_then_children"),
    "tournament": ("mu_plus_lambda_tournament", "first_drawn"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("depsign",), default="depsign")
    parser.add_argument("--search-space", choices=("components",), default="components")
    parser.add_argument("--survivors", choices=POLICIES, default="elite")
    parser.add_argument("--label-mode", choices=("binary", "three_class"), default="three_class")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--generations", type=run_ga.positive_integer)
    parser.add_argument("--parent-selection", choices=run_ga.PARENT_SELECTION_MODES)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--config", type=Path, default=run_ga.CONFIG,
                        help="Base GA configuration; only survivor selection and its tie rule are overridden.")
    parser.add_argument("--output-dir", type=Path,
                        help="New trial directory containing effective_config.json and the full GA results in ga/.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Accepted for production CLI compatibility; component search always runs the requested generations.")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    mode, tie_rule = POLICIES[args.survivors]
    config["search"].update(survivor_selection=mode, tie_break=tie_rule)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trial = args.output_dir or ROOT / "experiment/runs/components" / f"{args.label_mode}_{args.survivors}_{args.seed}_{stamp}"
    trial.mkdir(parents=True, exist_ok=False)
    snapshot = trial / "effective_config.json"
    snapshot.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Component survivor selection: {mode}", flush=True)
    result = component_ga.run_components(
        trial / "ga", config_file=snapshot.resolve(), label_mode=args.label_mode,
        data_file=args.data_file, seed=args.seed, generations=args.generations,
        parent_selection=args.parent_selection, prepare_only=args.prepare_only,
    )
    return 0 if result["status"] in {"prepared", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
