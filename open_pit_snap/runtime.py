"""Bind project settings into notebook-style module globals."""

from __future__ import annotations

from pathlib import Path

from . import logic
from .config import Project, resolve_gee_project, resolve_token


def bind(project: Project, *, earthdata_token: Path | None = None, ee_project: str | None = None) -> None:
    settings = project.settings
    ref = settings["reference"]
    optical = settings.get("optical") or {}
    quality = settings.get("quality") or {}
    pairs = settings.get("pairs") or {}

    logic.START_DATE = settings["period"]["start"]
    logic.END_DATE = settings["period"]["end"]
    logic.STABLE_POINTS_LONGITUDE = float(ref["center"][0])
    logic.STABLE_POINTS_LATITUDE = float(ref["center"][1])
    logic.VOIVODESHIP = ref["voivodeship"]
    logic.EARTHDATA_TOKEN = resolve_token(earthdata_token)
    logic.GEE_PROJECT_NAME = optical.get("ee_project") or resolve_gee_project(ee_project)
    logic.MINE_NAME = settings.get("name") or project.root.name
    logic.MINIMUM_SCENES = int(quality.get("minimum_acquisitions", 5))
    logic.MAX_PAIR_DAYS = int(pairs.get("max_days", 13))
    logic.MIN_TEMPORAL_COVERAGE = float(ref.get("min_temporal_coverage", 0.8))
    logic.MIN_MEAN_COHERENCE = float(ref.get("min_mean_coherence", 0.8))
    logic.MAX_COHERENCE_STD = float(ref.get("max_coherence_std", 0.2))
    logic.MIN_CLUSTER_PIXELS = int(ref.get("min_cluster_pixels", 3))
    logic.MIN_STABLE_PIXELS = int(ref.get("min_stable_pixels", 30))

    nisar_cfg = settings.get("nisar") or {}
    if nisar_cfg:
        from . import nisar

        if "min_valid_coverage" in nisar_cfg:
            nisar.MIN_VALID_COVERAGE = float(nisar_cfg["min_valid_coverage"])
        boxcar = nisar_cfg.get("boxcar")
        if boxcar is not None and len(boxcar) == 2:
            nisar.WINDOW_RG = int(boxcar[0])
            nisar.WINDOW_AZ = int(boxcar[1])

    raw = project.root / "raw"
    coherence = project.root / "coherence"
    maps = project.root / "activity_maps"
    for path in (raw, coherence, maps):
        path.mkdir(parents=True, exist_ok=True)
    logic.DOWNLOAD_DIR = raw
    logic.OUTPUT_DIR = maps
    logic.COHERENCE_DIR = coherence
    logic.POLYGON_PATH = project.mine_path

    logic.AOI_MINE = logic.make_polygon_buffer(str(project.mine_path))
    logic.AOI_STABLE_POINTS = logic.make_reference_window()
    logic.AOI_PROCESSING = logic.unary_union([
        logic.shapely_wkt.loads(logic.AOI_MINE),
        logic.shapely_wkt.loads(logic.AOI_STABLE_POINTS),
    ]).wkt
