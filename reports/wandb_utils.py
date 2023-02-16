from collections import defaultdict
from typing import Optional

import wandb

api = wandb.Api()


def get_runs(
    project_name: str, filters: Optional[dict[str, str]] = None
) -> list[wandb.apis.public.Run]:
    runs = api.runs(project_name, filters=filters)
    return [run for run in runs]


def get_run(project_name: str, id: str = None) -> wandb.apis.public.Runs:
    run = api.run(f"{project_name}/{id}")
    return run


def get_run_ids(
    project_name: str, filters: Optional[dict[str, str]] = None
) -> list[str]:
    runs = get_runs(project_name, filters=filters)
    run_ids = [run.id for run in runs]
    return run_ids


def get_best_runs(
    project_name: str, metric_name: str, filters: Optional[dict[str, str]] = None
) -> dict[str, wandb.apis.public.Run]:
    runs = get_runs(project_name, filters=filters)
    # get the best run for each group
    best_runs = defaultdict(lambda: None)

    for run in runs:
        group = run.config["model"]["name"]
        if (
            best_runs[group] is None
            or run.summary["test"]["all"][metric_name]
            > best_runs[group].summary["test"]["all"][metric_name]
        ):
            best_runs[group] = run
    return best_runs
