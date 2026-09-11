#!/usr/bin/env python3
"""Publication-ready Zambia DHS 2024 analysis orchestrator.

Run order:
1. core primary 60/20/20 model-development analysis;
2. repeated grouped capacity selection on the frozen 80% development sample;
3. publication extensions (outer stability, domain analyses, calibration diagnostics);
4. locked-result verification.

DHS microdata are never written to the repository. They must be supplied locally
under authorised DHS access.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "analysis_config.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the publication reproducibility pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--full-retune", action="store_true", help="Run the complete candidate-model grid search/SHAP core pipeline instead of the fast locked-primary reproduction.")
    parser.add_argument("--full-stability", action="store_true", help="Refit all comparator models across the five stability folds (slower).")
    parser.add_argument("--include-stacked-reconstruction", action="store_true", help="Run the best-effort stacked-domain reconstruction; excluded from locked verification because the exact original stacked implementation was not recovered.")
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-extensions", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    ir, mr, ar = [PROJECT_ROOT / paths[k] for k in ("ir_path", "mr_path", "ar_path")]
    out = PROJECT_ROOT / paths["output_dir"]
    core_out = out / "core"
    cap_out = out / "capacity_selection"
    ext_out = out / "publication_extensions"
    out.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in (ir, mr, ar) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing authorised DHS file(s):\n- " + "\n- ".join(missing))

    if not args.skip_core:
        if args.full_retune:
            run([sys.executable, str(SRC_ROOT / "core_model_pipeline.py"), "--ir", str(ir), "--mr", str(mr), "--ar", str(ar), "--output", str(core_out)])
        else:
            run([sys.executable, str(SRC_ROOT / "publication_primary_run.py"), "--config", str(args.config), "--output", str(core_out)])
    if not args.skip_capacity:
        run([sys.executable, str(SRC_ROOT / "capacity_selection.py"), "--config", str(args.config), "--core-output", str(core_out), "--output", str(cap_out)])
    if not args.skip_extensions:
        ext_cmd = [sys.executable, str(SRC_ROOT / "publication_extensions.py"), "--config", str(args.config), "--core-output", str(core_out), "--output", str(ext_out)]
        if args.full_stability:
            ext_cmd.append("--full-stability")
        if args.include_stacked_reconstruction:
            ext_cmd.append("--include-stacked-reconstruction")
        run(ext_cmd)
    if not args.skip_verify:
        run([sys.executable, str(SRC_ROOT / "verify_locked_results.py")])

    manifest = {
        "status": "completed",
        "config": args.config.name,
        "analytical_contract": "capacity selected on 80% development; frozen primary test probabilities retained; no 80% final-test refit",
        "primary_seed": cfg["random_states"]["primary_split_and_tuning"],
        "five_fold_stability_seed": cfg["random_states"]["five_fold_model_stability"],
        "capacity_seeds": cfg["random_states"]["repeated_capacity_validation"],
        "selected_nominal_capacity": cfg["capacity_selection"]["selected_nominal_capacity"],
        "software": cfg["software"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
