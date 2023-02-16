import shutil 
import os
from pathlib import Path
from reports.wandb_utils import get_best_runs
from src.settings import PROJECT, EXPERIMENT_DIR, MODEL_NAMES

def copy_model(best_runs: dict, dataset:str):
    for model_name, run in best_runs.items():
        if model_name in MODEL_NAMES:
            model_name = MODEL_NAMES[model_name]
        from_path = experiment_dir / run.id
        to_path = Path(f"files/model_checkpoints/{dataset}") / model_name.lower()
        to_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            from_path/"final_model.pt",
            to_path,
        )
        shutil.copy(
            from_path/"config.yaml",
            to_path,
        )
        shutil.copy(
            from_path/"target2index.json",
            to_path,
        )
        if os.path.exists(from_path/"token2index.json"):
            shutil.copy(
                from_path/"token2index.json",
                to_path,
            )


mimiciv_icd9_sweepid = "5ykjry46"
mimiciv_icd10_sweepid = "46ah61d0"
mimiciii_clean_sweepid = "83jehgfy"
experiment_dir = Path(EXPERIMENT_DIR)

best_runs = runs = get_best_runs(
    PROJECT, "f1_micro", {"Sweep": {"$regex": f"{mimiciv_icd9_sweepid}"}, "State": "finished"}
)
copy_model(best_runs, "mimiciv_icd9")

best_runs = runs = get_best_runs(
    PROJECT, "f1_micro", {"Sweep": {"$regex": f"{mimiciv_icd10_sweepid}"}, "State": "finished"}
)
copy_model(best_runs, "mimiciv_icd10")

best_runs = runs = get_best_runs(
    PROJECT, "f1_micro", {"Sweep": {"$regex": f"{mimiciii_clean_sweepid}"}, "State": "finished"}
)
copy_model(best_runs, "mimiciii_clean")
