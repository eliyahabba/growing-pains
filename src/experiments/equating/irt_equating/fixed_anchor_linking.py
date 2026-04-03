from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from irt import TrainingConfig, train_item_parameters, save_item_parameters, select_anchors_structured_with_matrix, save_anchors_structured


def _load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrix not found: {path}")
    return pd.read_parquet(path)


def _build_anchor_items(
    baseline_params: pd.DataFrame, 
    available_questions: set[str],
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> list[dict]:
    """Build anchor items from baseline parameters.
    
    If A_matrix and B_matrix are provided (full MIRT vectors), uses them directly
    as vector anchors. Otherwise, falls back to scalar anchors (a, b columns).
    
    Args:
        baseline_params: DataFrame with 'a' and 'b' columns, indexed by question_id
        available_questions: Set of question IDs available in the current dataset
        A_matrix: Full discrimination matrix, shape (1, D, n_items)
        B_matrix: Full difficulty matrix, shape (1, D, n_items)
    
    Returns:
        List of anchor item dicts with either vector or scalar parameters
    """
    baseline_params = baseline_params.copy()
    baseline_params.index = baseline_params.index.astype(str)
    subset = baseline_params.loc[baseline_params.index.intersection(available_questions)]
    if subset.empty:
        raise ValueError("No overlap between baseline item params and current matrix for anchoring")
    
    # Get the question order from baseline_params index
    baseline_qids = list(baseline_params.index)
    
    anchors = []
    for item_id in subset.index:
        anchor = {"item_id": item_id}
        
        # Try to use vector parameters if available
        if A_matrix is not None and B_matrix is not None:
            try:
                base_idx = baseline_qids.index(item_id)
                # Extract vectors: A_matrix shape is (1, D, n_items)
                anchor["discrimination_vector"] = A_matrix[0, :, base_idx].tolist()
                anchor["difficulty_vector"] = B_matrix[0, :, base_idx].tolist()
            except (ValueError, IndexError):
                # Fall back to scalar if vector extraction fails
                anchor["difficulty"] = float(subset.loc[item_id, "b"])
                anchor["discrimination"] = float(subset.loc[item_id, "a"])
        else:
            # Use scalar parameters
            anchor["difficulty"] = float(subset.loc[item_id, "b"])
            anchor["discrimination"] = float(subset.loc[item_id, "a"])
        
        anchors.append(anchor)
    
    return anchors


def run_fixed_anchor_linking(
    skill: str,
    skills_root: Path,
    output_subdir: str = "equating/fixed_anchor",
    number_item_per_scenario: int = 100,
    dims_search: str = "5,10",
    device: str | None = None,
    epochs: int = 2000,
    lr: float = 0.01,
    skip_existing: bool = True,
) -> Path | None:
    skill_dir = skills_root / skill
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    output_dir = skill_dir / output_subdir
    out_path = output_dir / "item_params_fixed_anchor.parquet"
    
    # Skip if results already exist
    if skip_existing and out_path.exists():
        print(f"   → Results already exist, skipping: {out_path}")
        return None

    train_df = _load_matrix(skill_dir / "matrix_train_base.parquet")
    link_df = _load_matrix(skill_dir / "matrix_train_link.parquet")
    test_df = _load_matrix(skill_dir / "matrix_test_base.parquet")
    baseline_params = pd.read_parquet(skill_dir / "irt" / "item_params.parquet")
    
    # Load baseline MIRT matrices if available
    baseline_meta_path = skill_dir / "irt" / "item_params.meta.json"
    A_baseline, B_baseline = None, None
    if baseline_meta_path.exists():
        try:
            with open(baseline_meta_path) as f:
                baseline_meta = json.load(f)
            if "A_matrix" in baseline_meta and "B_matrix" in baseline_meta:
                A_baseline = np.array(baseline_meta["A_matrix"])
                B_baseline = np.array(baseline_meta["B_matrix"])
                print(f"   ✓ Loaded baseline MIRT matrices: A{A_baseline.shape}, B{B_baseline.shape}")
        except Exception as e:
            print(f"   ⚠ Could not load baseline matrices: {e}")

    combined_df = pd.concat([train_df, link_df], ignore_index=True).drop_duplicates()
    available_questions = set(combined_df["question_id"].astype(str).unique())
    anchor_items = _build_anchor_items(baseline_params, available_questions, A_baseline, B_baseline)
    
    # Log anchor type
    if anchor_items and "discrimination_vector" in anchor_items[0]:
        print(f"   ✓ Using VECTOR anchors for {len(anchor_items)} items")
    else:
        print(f"   ⚠ Using SCALAR anchors for {len(anchor_items)} items (vectors not available)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target dimension - MUST match anchor vectors if using vector anchors
    target_dims = [int(d.strip()) for d in dims_search.split(",") if d.strip()]
    
    # Priority 1: Get dimension directly from A_baseline matrix shape (most reliable)
    if A_baseline is not None:
        baseline_dim = A_baseline.shape[1] if A_baseline.ndim == 3 else A_baseline.shape[0]
        target_dims = [baseline_dim]
        print(f"   ✓ Using dimension {baseline_dim} from baseline A_matrix shape")
    # Priority 2: Fallback to attrs if available
    elif hasattr(baseline_params, "attrs") and "best_dimension" in baseline_params.attrs:
        baseline_dim = int(baseline_params.attrs["best_dimension"])
        target_dims = [baseline_dim]
        print(f"   ✓ Using dimension {baseline_dim} from baseline attrs")
    elif len(target_dims) > 1:
        print(f"   ⚠ Warning: No baseline dimension found, running search {target_dims}.")
        print(f"     This may fail if anchor vectors have a different dimension!")

    # Build config kwargs, omitting device if None to use auto-detection
    config_kwargs = {
        "number_item_per_scenario": number_item_per_scenario,
        "dims_search": target_dims,
        "epochs": epochs,
        "lr": lr,
    }
    if device is not None:
        config_kwargs["device"] = device
    
    cfg = TrainingConfig(**config_kwargs)

    params = train_item_parameters(
        combined_df,
        test_matrix_df=test_df,
        config=cfg,
        output_dir=str(output_dir),
        anchor_items=anchor_items,
    )

    # Note: With vector anchors, py-irt now preserves the exact vectors during training,
    # so we no longer need to manually replace them after training.

    out_path = output_dir / "item_params_fixed_anchor.parquet"
    save_item_parameters(params, str(out_path))

    combined_path = output_dir / "matrix_train_link.parquet"
    combined_df.to_parquet(combined_path, index=False)

    # Select Equated Anchors (Base+Link)
    try:
        anchors, weights = select_anchors_structured_with_matrix(
            params, combined_df, number_items=number_item_per_scenario, method="irt_clustering"
        )
        anchors_path = output_dir / f"anchors_fixed_{number_item_per_scenario}.json"
        save_anchors_structured(anchors, weights, str(anchors_path))
        print(f"   ✓ Selected {number_item_per_scenario} equated anchors saved to {anchors_path}")
    except Exception as e:
        print(f"   ⚠ Failed to select fixed anchors: {e}")

    metadata = {
        "skill": skill,
        "total_questions": len(combined_df["question_id"].unique()),
        "anchor_count": len(anchor_items),
        "output_path": str(out_path),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out_path


def _process_skill(kwargs: dict) -> tuple[str, str, Exception | None]:
    """Worker function for parallel processing. Returns (skill, status, error)."""
    skill = kwargs.pop("skill")
    try:
        out_path = run_fixed_anchor_linking(skill=skill, **kwargs)
        if out_path:
            return skill, "completed", None
        else:
            return skill, "skipped", None
    except Exception as e:
        return skill, "error", e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-anchor calibration with py-irt anchor items support")
    parser.add_argument("--skill", default="Entailment & Bias", help="Skill name (directory under skills root). If not provided, runs on all skills.")
    parser.add_argument(
        "--skills-root",
        default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "skills"),
        help="Root directory containing per-skill artifacts",
    )
    parser.add_argument("--number-item-per-scenario", type=int, default=100)
    parser.add_argument("--dims-search", default="5,10", help="Comma-separated list of dimensions to search")
    parser.add_argument("--device", default=None, help="Device for training (cpu/cuda). Auto-detects if not specified.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="Force rerun even if results already exist", default=True)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers. Use 1 for sequential processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_root = Path(args.skills_root)
    
    if args.skill:
        # Single skill mode
        skills = [args.skill]
    else:
        # All skills mode
        if not skills_root.exists():
            print(f"Skills root not found: {skills_root}")
            return
        skills = [d.name for d in skills_root.iterdir() if d.is_dir()]
        if not skills:
            print(f"No skills found in {skills_root}")
            return
        print(f"Running fixed-anchor calibration on {len(skills)} skills: {', '.join(skills)}")
        print(f"Using {args.workers} worker(s)\n")
    
    # Build common kwargs for all skills
    common_kwargs = {
        "skills_root": skills_root,
        "number_item_per_scenario": args.number_item_per_scenario,
        "dims_search": args.dims_search,
        "device": args.device,
        "epochs": args.epochs,
        "lr": args.lr,
        "skip_existing": not args.force,
    }
    
    if args.workers == 1:
        # Sequential processing (original behavior)
        for skill in skills:
            try:
                print(f"Processing skill: {skill}")
                out_path = run_fixed_anchor_linking(skill=skill, **common_kwargs)
                if out_path:
                    print(f"✓ [{skill}] Fixed-anchor calibration complete. Item params saved to {out_path}\n")
                else:
                    print(f"⊘ [{skill}] Skipped (results already exist)\n")
            except Exception as e:
                print(f"✗ [{skill}] Error: {e}\n")
    else:
        # Parallel processing
        task_kwargs_list = [{"skill": skill, **common_kwargs} for skill in skills]
        
        completed = 0
        total = len(skills)
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_skill, kwargs): kwargs["skill"] for kwargs in task_kwargs_list}
            
            for future in as_completed(futures):
                skill = futures[future]
                completed += 1
                try:
                    skill_name, status, error = future.result()
                    if status == "completed":
                        print(f"✓ [{completed}/{total}] [{skill_name}] Fixed-anchor calibration complete")
                    elif status == "skipped":
                        print(f"⊘ [{completed}/{total}] [{skill_name}] Skipped (results already exist)")
                    else:
                        print(f"✗ [{completed}/{total}] [{skill_name}] Error: {error}")
                except Exception as e:
                    print(f"✗ [{completed}/{total}] [{skill}] Unexpected error: {e}")


if __name__ == "__main__":
    main()

