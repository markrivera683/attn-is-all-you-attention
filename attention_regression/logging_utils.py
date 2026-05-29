import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb


def make_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}-{timestamp}"


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


class ExperimentLogger:
    def __init__(
        self,
        run_dir: Path,
        config: Any,
        *,
        use_wandb: bool = False,
        wandb_project: str = "attention mechanism",
        wandb_name: str | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_json_path = self.run_dir / "summary.json"
        self.summary_csv_path = self.run_dir / "summary.csv"
        self.use_wandb = use_wandb
        self._wandb_run = None

        (self.run_dir / "config.json").write_text(
            json.dumps(to_jsonable(config), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.metrics_file = self.metrics_path.open("a", encoding="utf-8")

        if self.use_wandb:
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name or self.run_dir.name,
                config=to_jsonable(config),
                dir=str(self.run_dir),
            )

    def log(self, values: dict[str, Any], *, step: int, phase: str) -> None:
        row = {
            "step": step,
            "phase": phase,
            **to_jsonable(values),
        }
        self.metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        self.metrics_file.flush()
        if self.use_wandb:
            wandb.log(values, step=step)

    def write_summary(self, summaries: dict[str, dict[str, Any]]) -> None:
        json_ready = to_jsonable(summaries)
        self.summary_json_path.write_text(
            json.dumps(json_ready, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        fieldnames = sorted(
            {
                key
                for summary in json_ready.values()
                for key in summary.keys()
            }
            | {"model"}
        )
        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for model, summary in sorted(json_ready.items()):
                writer.writerow({"model": model, **summary})

        if self.use_wandb:
            for model, summary in summaries.items():
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        wandb.summary[f"{model}/{key}"] = value

    def close(self) -> None:
        if not self.metrics_file.closed:
            self.metrics_file.close()
        if self.use_wandb:
            wandb.finish()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
