"""Project settings for the notebook-faithful SNAP pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

DEFAULTS = {
    "schema_version": 1,
    "grid": {"mine_buffer_m": 200},
    "pairs": {"min_days": 1, "max_days": 13, "exclude_months": [12, 1, 2]},
    "reference": {
        "half_width_m": 1500,
        "min_pairs": 4,
        "min_temporal_coverage": 0.8,
        "min_mean_coherence": 0.8,
        "max_coherence_std": 0.2,
        "min_cluster_pixels": 3,
        "min_stable_pixels": 30,
    },
    "quality": {"minimum_acquisitions": 5},
    "optical": {"source": "earth-engine", "ee_project": None},
    # auto | sentinel1 | nisar — auto picks NISAR when period.start >= 2026-06-17
    "sensor": "auto",
}


@dataclass
class Project:
    root: Path
    settings: dict

    @property
    def start(self) -> date:
        return date.fromisoformat(self.settings["period"]["start"])

    @property
    def end(self) -> date:
        return date.fromisoformat(self.settings["period"]["end"])

    @property
    def mine_path(self) -> Path:
        path = Path(self.settings["mine"])
        return path if path.is_absolute() else (self.root / path).resolve()

    @classmethod
    def load(cls, root: Path) -> "Project":
        path = root / "project.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid project.yaml: {path}")
        settings = copy.deepcopy(DEFAULTS)
        for key, value in raw.items():
            if key in settings and isinstance(settings[key], dict) and isinstance(value, dict):
                settings[key] = {**settings[key], **value}
            else:
                settings[key] = value
        for required in ("name", "mine", "period", "reference"):
            if required not in settings:
                raise ValueError(f"project.yaml missing {required}")
        return cls(root.resolve(), settings)


def save_project(root: Path, settings: dict) -> Path:
    path = root / "project.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return path


def resolve_token(path: Path | None = None) -> str:
    candidates = []
    if path:
        candidates.append(Path(path))
    here = Path(__file__).resolve().parents[1]
    candidates.extend([
        here / "token_asf",
        Path(r"c:\space_technologies\Kopalnie\open-pit-baseline\token_asf"),
        Path(r"c:\space_technologies\Kopalnie\open-pit-mining\sar\token_asf"),
        Path.home() / "token_asf",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Earthdata token not found (token_asf)")


def resolve_sensor(settings: dict) -> str:
    """Return ``nisar`` or ``sentinel1`` for this project period."""
    from .nisar import prefer_nisar

    requested = str(settings.get("sensor") or "auto").lower()
    if requested in ("nisar", "sentinel1"):
        return requested
    if requested != "auto":
        raise ValueError("sensor must be auto, nisar, or sentinel1")
    return "nisar" if prefer_nisar(settings["period"]) else "sentinel1"


def resolve_gee_project(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    here = Path(__file__).resolve().parents[1]
    for candidate in (
        here / "gee_project",
        Path(r"c:\space_technologies\Kopalnie\open-pit-baseline\gee_project"),
        Path(r"c:\space_technologies\Kopalnie\open-pit-mining\sar\gee_project"),
        Path.home() / "gee_project",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Earth Engine project id not found (gee_project)")
