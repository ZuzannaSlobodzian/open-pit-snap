"""Orchestrate catalog → download → SNAP coherence → NDAI activity maps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from . import logic
from .config import Project, resolve_sensor
from .runtime import bind


def _write_coherence_geotiff(pair: dict, kind: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    master = pair["master_day"].strftime("%Y%m%d")
    slave = pair["slave_day"].strftime("%Y%m%d")
    path = directory / f"{master}_{slave}_{kind}_coherence.tif"
    data = np.asarray(pair["coherence"], dtype=np.float32)
    transform = pair["transform"]
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "coherence")
        if pair.get("shadow") is not None:
            pass
    if pair.get("shadow") is not None:
        shadow_path = directory / f"{master}_{slave}_{kind}_layover_shadow.tif"
        shadow = np.asarray(pair["shadow"], dtype=np.float32)
        with rasterio.open(
            shadow_path, "w", driver="GTiff", height=shadow.shape[0], width=shadow.shape[1],
            count=1, dtype="float32", crs="EPSG:4326", transform=transform, compress="deflate",
        ) as dst:
            dst.write(shadow, 1)
    return path


def catalog(project: Project):
    """Search ASF (in memory only — nothing written to disk)."""
    sensor = resolve_sensor(project.settings)
    if sensor == "nisar":
        from . import nisar

        scenes, pin = nisar.find_nisar_scenes()
        print(f"Catalog: {len(scenes)} NISAR GSLC scene(s) (not saved to disk)", flush=True)
        return scenes, pin
    scenes = logic.find_sentinel_1_scenes()
    print(f"Catalog: {len(scenes)} SAFE scene(s) (not saved to disk)", flush=True)
    return scenes, None


def download(project: Project, scenes=None) -> list[Path]:
    sensor = resolve_sensor(project.settings)
    if sensor == "nisar":
        from . import nisar

        found, found_pin = nisar.find_nisar_scenes()
        scenes = scenes if scenes is not None else found
        token = logic.EARTHDATA_TOKEN
        return [nisar.download_gslc(scene, found_pin, token) for scene in scenes]

    if scenes is None:
        scenes = logic.find_sentinel_1_scenes()
    session = logic.start_asf_session()
    paths = []
    try:
        for scene in scenes:
            paths.append(logic.download_images(scene, session))
    finally:
        logic.close_asf_session(session)
    return paths


def coherence(project: Project, scenes=None, pin=None) -> tuple[list, list]:
    sensor = resolve_sensor(project.settings)
    if sensor == "nisar":
        from . import nisar

        if scenes is None or pin is None:
            scenes, pin = nisar.find_nisar_scenes()
        mine_stack, ref_stack = nisar.coherence_stacks(scenes, pin, logic.EARTHDATA_TOKEN)
        for pair in mine_stack:
            _write_coherence_geotiff(pair, "mine", logic.COHERENCE_DIR / "mine")
        for pair in ref_stack:
            _write_coherence_geotiff(pair, "reference", logic.COHERENCE_DIR / "reference")
        print(f"Coherence pairs: {len(mine_stack)} → {logic.COHERENCE_DIR}", flush=True)
        return mine_stack, ref_stack

    if scenes is None:
        scenes = logic.find_sentinel_1_scenes()
    session = logic.start_asf_session()
    mine_stack, ref_stack = [], []
    try:
        master = None
        for index in range(len(scenes) - 1):
            if master is None:
                master = logic.download_images(scenes[index], session)
            slave = logic.download_images(scenes[index + 1], session)
            print(
                "Pair %d/%d  %s  x  %s"
                % (index + 1, len(scenes) - 1, master.name, slave.name),
                flush=True,
            )
            gap = (logic._scene_day(slave) - logic._scene_day(master)).days
            if gap <= logic.MAX_PAIR_DAYS:
                pair = logic.coherence_analysis(master, slave)
                mine_stack.append(pair["mine"])
                ref_stack.append(pair["stable_points"])
                _write_coherence_geotiff(pair["mine"], "mine", logic.COHERENCE_DIR / "mine")
                _write_coherence_geotiff(pair["stable_points"], "reference", logic.COHERENCE_DIR / "reference")
            else:
                print(
                    "  skip pair, %d days between scenes, pair is valid when gap is %d days or less"
                    % (gap, logic.MAX_PAIR_DAYS),
                    flush=True,
                )
            # logic.remove_scene(master)
            master = slave
    finally:
        logic.close_asf_session(session)

    print(f"Coherence pairs: {len(mine_stack)} → {logic.COHERENCE_DIR}", flush=True)
    return mine_stack, ref_stack


def _read_pair_geotiff(path: Path, master: str, slave: str) -> dict:
    with rasterio.open(path) as src:
        coherence = src.read(1).astype(np.float32)
        transform = src.transform
        height, width = src.height, src.width
    shadow_path = path.with_name(path.name.replace("_coherence.tif", "_layover_shadow.tif"))
    shadow = None
    if shadow_path.is_file():
        with rasterio.open(shadow_path) as src:
            shadow = src.read(1).astype(np.float32)
    left, top = transform.c, transform.f
    dx, dy = transform.a, transform.e
    extent = (left, left + dx * width, top + dy * height, top)
    return {
        "master_day": datetime.strptime(master, "%Y%m%d"),
        "slave_day": datetime.strptime(slave, "%Y%m%d"),
        "coherence": coherence,
        "shadow": shadow,
        "transform": transform,
        "extent": extent,
    }


def load_coherence_stacks(project: Project) -> tuple[list, list]:
    """Reload mine/reference coherence arrays saved by a prior coherence run."""
    mine_dir = project.root / "coherence" / "mine"
    ref_dir = project.root / "coherence" / "reference"
    mine_files = sorted(mine_dir.glob("*_mine_coherence.tif"))
    if not mine_files:
        raise FileNotFoundError(f"No coherence GeoTIFFs under {mine_dir}. Run coherence first.")
    mine_stack, ref_stack = [], []
    for path in mine_files:
        # 20240302_20240314_mine_coherence.tif
        master, slave, *_ = path.name.split("_")
        mine_stack.append(_read_pair_geotiff(path, master, slave))
        ref_path = ref_dir / f"{master}_{slave}_reference_coherence.tif"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Missing reference coherence: {ref_path}")
        ref_stack.append(_read_pair_geotiff(ref_path, master, slave))
    print(f"Loaded {len(mine_stack)} coherence pair(s) from coherence/", flush=True)
    return mine_stack, ref_stack


def run_activity(project: Project, mine_stack: list | None = None, ref_stack: list | None = None) -> int:
    if mine_stack is None or ref_stack is None:
        mine_stack, ref_stack = load_coherence_stacks(project)
    gamma = logic.stable_points_analysis(ref_stack)
    print("Calculating Normalized Differential Activity Index treshold ...", flush=True)
    by_pair, extent, grid_transform, shape_hw, mine, mine_mask = logic.coherence_mine_grid(mine_stack, gamma)
    prepared, samples = [], []
    for index, reference in enumerate(gamma, 1):
        master_day = reference["master_day"]
        slave_day = reference["slave_day"]
        print(
            "Pair %d/%d: %s - %s"
            % (index, len(gamma), master_day.strftime("%Y.%m.%d"), slave_day.strftime("%Y.%m.%d")),
            flush=True,
        )
        image = logic.find_sentinel2_images(
            slave_day + logic.timedelta(days=1),
            logic._extend_date_range_by_month(slave_day, 1),
            first_only=True,
        )
        if not image:
            print(
                "  skip %s - %s, no cloud-free Sentinel-2 scene within one month after"
                % (master_day.strftime("%Y.%m.%d"), slave_day.strftime("%Y.%m.%d")),
                flush=True,
            )
            continue
        print("    Sentinel-2 scene selected: %s" % logic._image_label(image), flush=True)
        logic.ndai_samples(
            image, reference, by_pair, extent, shape_hw, grid_transform, mine_mask,
            master_day, slave_day, samples, prepared,
        )
    threshold = logic.ndai_threshold(samples)
    print("Calculating Normalized Differential Activity Index maps ...", flush=True)
    mapped = 0
    for number, pair in enumerate(prepared, 1):
        print(
            "Pair %d/%d: %s - %s"
            % (number, len(prepared), pair["master_day"].strftime("%Y.%m.%d"), pair["slave_day"].strftime("%Y.%m.%d")),
            flush=True,
        )
        change = logic.activity_mask(pair, threshold)
        area, percent = logic.changed_area_inside_polygon(change, mine_mask, grid_transform)
        logic.activity_map_with_masks(
            change.astype(np.uint8), mine, pair["master_day"], pair["slave_day"], extent, area, percent,
            pair["water"], pair["vegetation"], pair["shadow"], pair["cloud"],
        )
        logic.activity_map(
            change.astype(np.uint8), mine, pair["master_day"], pair["slave_day"], extent, area, percent,
        )
        logic.activity_map_with_sentinel2_references(
            change.astype(np.uint8), mine, pair["master_day"], pair["slave_day"], extent, area, percent,
        )
        mapped += 1
    print(f"Activity maps written under {logic.OUTPUT_DIR} ({mapped} pairs)", flush=True)
    return mapped


def pipeline(project: Project, *, earthdata_token=None, ee_project=None) -> int:
    bind(project, earthdata_token=earthdata_token, ee_project=ee_project)
    sensor = resolve_sensor(project.settings)
    print(f"sensor: {sensor}", flush=True)
    print("catalog …", flush=True)
    scenes, pin = catalog(project)
    if sensor == "nisar":
        print("coherence (NISAR GSLC subset + boxcar, no SAFE/SNAP) …", flush=True)
    else:
        print("coherence (download SAFE + SNAP) …", flush=True)
    mine_stack, ref_stack = coherence(project, scenes, pin)
    if not mine_stack:
        raise RuntimeError("No coherence pairs were produced")
    print("activity maps …", flush=True)
    mapped = run_activity(project, mine_stack, ref_stack)
    return 0 if mapped else 1
