"""NISAR L-band GSLC path: already-geocoded SLCs → boxcar coherence (no SAFE/SNAP)."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import asf_search as asf
import h5py
import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from shapely import wkt as shapely_wkt
from shapely.geometry import shape
from shapely.ops import unary_union

from . import logic

NISAR_PUBLIC_START = "2026-06-17"
NISAR_VALIDATED_START = "2026-10-21"
NISAR_GRIDS = "/science/LSAR/GSLC/grids"
NISAR_NAME = re.compile(
    r"NISAR_L2_[A-Z]+_GSLC_\d{3}_(?P<track>\d{3})_(?P<direction>[AD])_(?P<frame>\d{3})_"
    r"(?P<mode>[A-Z0-9]{4})_[A-Z0-9]+_(?P<source>[A-Z])_(?P<day>\d{8})T\d{6}.*_\d{3}\.h5"
)
WINDOW_RG, WINDOW_AZ = 10, 3  # SNAP-style rg × az; overridable via project nisar.boxcar
MIN_VALID_COVERAGE = 0.5  # drop GSLC if finite mask covers less of AOI; overridable


def prefer_nisar(period: dict) -> bool:
    return str(period["start"]) >= NISAR_PUBLIC_START


def short_name(period: dict) -> str:
    if str(period["start"]) >= NISAR_VALIDATED_START:
        return "NISAR_L2_GSLC_V1"
    return "NISAR_L2_GSLC_PROVISIONAL_V1"


def collections_for_period(period: dict) -> list[str]:
    """Provisional before validation start; validated after; both when the window spans them."""
    start = str(period["start"])
    end = str(period.get("end") or period["start"])
    names: list[str] = []
    if start < NISAR_VALIDATED_START:
        names.append("NISAR_L2_GSLC_PROVISIONAL_V1")
    if end >= NISAR_VALIDATED_START:
        names.append("NISAR_L2_GSLC_V1")
    return names or [short_name(period)]


def _day(props) -> date:
    return date.fromisoformat(str(props.get("startTime", ""))[:10])


def _product_name(product) -> str:
    props = product.properties
    for key in ("fileName", "sceneName"):
        name = props.get(key)
        if name:
            return Path(str(name)).name
    url = props.get("url") or ""
    return Path(unquote(urlsplit(url).path)).name


def _source_letter(name: str) -> str:
    text = name if name.lower().endswith(".h5") else f"{name}.h5"
    match = NISAR_NAME.fullmatch(Path(text).name)
    return match.group("source").upper() if match else ""


def _track_frame_direction(name: str, props) -> tuple[int | None, int | None, str]:
    text = name if name.lower().endswith(".h5") else f"{name}.h5"
    match = NISAR_NAME.fullmatch(Path(text).name)
    direction = str(props.get("flightDirection") or "").upper()
    if match:
        direction = "ASCENDING" if match.group("direction") == "A" else "DESCENDING"
        return int(match.group("track")), int(match.group("frame")), direction
    track = props.get("pathNumber") or props.get("track")
    frame = props.get("frameNumber") or props.get("frame")
    return (int(track) if track is not None else None,
            int(frame) if frame is not None else None,
            direction)


def find_nisar_scenes():
    """Search provisional/validated GSLC covering the processing AOI; pin one orbit."""
    period = {"start": logic.START_DATE, "end": logic.END_DATE}
    collections = collections_for_period(period)
    print("Searching for NISAR GSLC scenes ...", flush=True)
    print(f"  shortName={collections}", flush=True)
    results = asf.search(
        intersectsWith=logic.AOI_PROCESSING,
        shortName=collections,
        start=f"{logic.START_DATE}T00:00:00Z",
        end=f"{logic.END_DATE}T23:59:59Z",
    )
    aoi = shapely_wkt.loads(logic.AOI_PROCESSING)
    by_day: dict[date, list] = {}
    for product in results:
        name = _product_name(product)
        if _source_letter(name) not in ("", "A"):
            # Prefer science-mode source A; skip maturity/source M when A exists later.
            continue
        day = _day(product.properties)
        by_day.setdefault(day, []).append(product)

    options = []
    for day in sorted(by_day):
        groups: dict[tuple[str, int, int], list] = {}
        for product in by_day[day]:
            track, frame, direction = _track_frame_direction(_product_name(product), product.properties)
            if track is None or frame is None or not direction:
                continue
            groups.setdefault((direction, track, frame), []).append(product)
        for (direction, track, frame), group in groups.items():
            geoms = [shape(item.geometry) for item in group if item.geometry]
            if not geoms:
                continue
            union = unary_union(geoms)
            if not union.is_empty and (union.covers(aoi) or union.intersects(aoi)):
                options.append((0 if direction == "DESCENDING" else 1, day, track, frame, direction, group))
    if not options:
        raise RuntimeError("No NISAR GSLC products cover the processing AOI in this period")
    options.sort()
    _, pin_day, track, frame, direction, _ = options[0]

    stack = []
    for day in sorted(by_day):
        for product in by_day[day]:
            t, f, d = _track_frame_direction(_product_name(product), product.properties)
            if t == track and f == frame and d == direction and _source_letter(_product_name(product)) in ("", "A"):
                stack.append(product)
                break  # one product per day
    stack.sort(key=lambda item: _day(item.properties))
    print(f"  scenes found: {len(stack)}", flush=True)
    print(f"  date range: {logic.START_DATE} - {logic.END_DATE}", flush=True)
    print(f"  track={track}; frame={frame}; direction={direction}; pinned from {pin_day}", flush=True)
    for index, product in enumerate(stack, 1):
        print(f"  {index:2d}. {_day(product.properties)}  {_product_name(product)}", flush=True)
    if len(stack) < logic.MINIMUM_SCENES:
        raise RuntimeError(
            f"Need at least {logic.MINIMUM_SCENES} NISAR acquisitions — got {len(stack)}"
        )
    return stack, {"track": track, "frame": frame, "direction": direction, "frequency": "A", "polarization": "HH"}


def _open_remote_h5(url: str, token: str):
    import fsspec

    fs = fsspec.filesystem(
        "https",
        client_kwargs={"headers": {"Authorization": f"Bearer {token}"}},
    )
    stream = fs.open(url, mode="rb", cache_type="first")
    return h5py.File(stream, mode="r", fs_page_size=4 * 1024 * 1024, rdcc_nbytes=1024 * 1024 * 1000)


def _h5_affine_crs(dataset: h5py.Dataset):
    group = dataset.parent
    x = group["xCoordinates"][:]
    y = group["yCoordinates"][:]
    dx = float(group["xCoordinateSpacing"][()])
    dy = float(group["yCoordinateSpacing"][()])
    transform = Affine(dx, 0, float(x[0]) - dx / 2, 0, dy, float(y[0]) - dy / 2)
    crs = f"EPSG:{int(group['projection'][()])}"
    return transform, crs


def _bbox_window(transform: Affine, crs: str, wkt: str, pad: int = 16) -> Window:
    geom = shapely_wkt.loads(wkt)
    west, south, east, north = transform_bounds("EPSG:4326", crs, *geom.bounds, densify_pts=21)
    inverse = ~transform
    left, top = inverse * (west, north)
    right, bottom = inverse * (east, south)
    col = int(np.floor(min(left, right))) - pad
    row = int(np.floor(min(top, bottom))) - pad
    width = int(np.ceil(max(left, right))) + pad - col
    height = int(np.ceil(max(top, bottom))) + pad - row
    return Window(col, row, max(1, width), max(1, height))


def download_gslc(product, pin: dict, token: str) -> Path:
    """Download an AOI subset of a remote GSLC into raw/ (HDF5)."""
    try:
        from opera_utils.nisar._download import _extract_subset_from_h5
    except ImportError as exc:
        raise RuntimeError(
            "NISAR downloads need opera-utils[nisar]. "
            "pip install 'opera-utils[nisar]>=0.25.9,<0.26' h5py fsspec"
        ) from exc

    props = product.properties
    name = _product_name(product)
    if not name.lower().endswith(".h5"):
        name = f"{name}.h5"
    target = Path(logic.DOWNLOAD_DIR) / name
    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"  reuse {name}", flush=True)
        return target

    url = props.get("url")
    if not url:
        raise RuntimeError(f"{name}: catalog entry has no download URL")
    subdataset = f"{NISAR_GRIDS}/frequency{pin['frequency']}/{pin['polarization']}"
    print(f"Download NISAR subset: {name}", flush=True)
    Path(logic.DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nisar-", dir=logic.DOWNLOAD_DIR) as pending:
        temp = Path(pending) / name
        with _open_remote_h5(url, token) as source:
            if subdataset not in source:
                # Fall back to first available co-pol under frequency A.
                freq = source[f"{NISAR_GRIDS}/frequency{pin['frequency']}"]
                for pol in ("HH", "VV", "HV", "VH"):
                    if pol in freq:
                        pin = {**pin, "polarization": pol}
                        subdataset = f"{NISAR_GRIDS}/frequency{pin['frequency']}/{pol}"
                        break
                else:
                    raise RuntimeError(f"{name}: no usable polarization in GSLC")
            image = source[subdataset]
            transform, crs = _h5_affine_crs(image)
            window = _bbox_window(transform, crs, logic.AOI_PROCESSING, pad=32)
            full = Window(0, 0, image.shape[1], image.shape[0])
            col = max(0, int(window.col_off))
            row = max(0, int(window.row_off))
            width = min(int(window.width), image.shape[1] - col)
            height = min(int(window.height), image.shape[0] - row)
            if width < 1 or height < 1:
                raise RuntimeError(f"{name}: AOI does not intersect product grid")
            rows = slice(row, row + height)
            cols = slice(col, col + width)
            _extract_subset_from_h5(
                source, temp, rows, cols,
                frequency=pin["frequency"],
                polarizations=[pin["polarization"]],
                source_name=name,
            )
            mask = image.parent.get("mask")
            if mask is not None:
                with h5py.File(temp, "r+") as output:
                    group = output[image.parent.name]
                    if "mask" not in group:
                        group.create_dataset("mask", data=mask[rows, cols], compression="gzip")
        os.replace(temp, target)
    print(f"  saved {target.name} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    return target


def _read_complex(path: Path, pin: dict) -> tuple[np.ndarray, Affine, str]:
    subdataset = f"{NISAR_GRIDS}/frequency{pin['frequency']}/{pin['polarization']}"
    with h5py.File(path, "r") as container:
        if subdataset not in container:
            raise RuntimeError(f"{path.name}: missing {subdataset}")
        dataset = container[subdataset]
        data = dataset[...]
        if data.dtype.kind != "c":
            converted = np.empty(data.shape, dtype=np.complex64)
            converted.real, converted.imag = data["r"], data["i"]
            data = converted
        else:
            data = data.astype(np.complex64, copy=False)
        transform, crs = _h5_affine_crs(dataset)
        mask = dataset.parent.get("mask")
        if mask is not None:
            values = mask[...]
            data = data.copy()
            data[(values == 0) | (values == 255)] = np.nan + 1j * np.nan
    return data, transform, crs


def _boxcar_coherence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = (WINDOW_AZ, WINDOW_RG)
    area = WINDOW_RG * WINDOW_AZ
    required = max(3, int(np.ceil(area * 0.75)))
    valid = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    left = np.where(valid, a, 0).astype(np.complex128)
    right = np.where(valid, b, 0).astype(np.complex128)
    cross = left * np.conj(right)

    def average(values):
        return uniform_filter(values, size=size, mode="constant", cval=0.0)

    numerator = np.hypot(average(cross.real), average(cross.imag))
    denominator = np.sqrt(
        average(left.real ** 2 + left.imag ** 2).clip(min=0)
        * average(right.real ** 2 + right.imag ** 2).clip(min=0)
    )
    samples = np.rint(average(valid.astype(np.float64)) * area)
    good = valid & (samples >= required) & (denominator > 0)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[good] = np.minimum(1, (numerator[good] / denominator[good]).astype(np.float32))
    return out


def _to_wgs84_pair(coherence: np.ndarray, transform: Affine, crs: str, wkt: str,
                   master_day: date, slave_day: date) -> dict:
    """Warp a UTM coherence chip to geographic CRS for notebook activity maps."""
    geom = shapely_wkt.loads(wkt)
    west, south, east, north = geom.bounds
    # ~NISAR 5–10 m → use ~0.0001° (~10 m) geographic posting.
    res = 0.0001
    width = max(8, int(np.ceil((east - west) / res)))
    height = max(8, int(np.ceil((north - south) / res)))
    dst_transform = Affine(res, 0.0, west, 0.0, -res, north)
    destination = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=coherence,
        destination=destination,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    left, bottom, right, top = array_bounds(height, width, dst_transform)
    extent = (left, right, bottom, top)
    # Activity/S2 path expects datetime (same as SAFE geotiff reload).
    master_dt = datetime.combine(master_day, datetime.min.time())
    slave_dt = datetime.combine(slave_day, datetime.min.time())
    return {
        "master_day": master_dt,
        "slave_day": slave_dt,
        "coherence": destination,
        "shadow": None,
        "transform": dst_transform,
        "extent": extent,
    }


def coherence_stacks(scenes, pin: dict, token: str) -> tuple[list, list]:
    """Download GSLCs as needed and build consecutive-pair mine/reference stacks."""
    paths = []
    days = []
    for product in scenes:
        path = download_gslc(product, pin, token)
        # Drop acquisitions whose validity mask barely covers the mine (activity AOI).
        data, transform, crs = _read_complex(path, pin)
        day = _day(product.properties)

        def _coverage(wkt: str) -> float:
            window = _bbox_window(transform, crs, wkt, pad=0)
            col = max(0, int(window.col_off))
            row = max(0, int(window.row_off))
            width = min(int(window.width), data.shape[1] - col)
            height = min(int(window.height), data.shape[0] - row)
            chip = data[row:row + height, col:col + width]
            return float(np.isfinite(chip).mean()) if chip.size else 0.0

        mine_cov = _coverage(logic.AOI_MINE)
        proc_cov = _coverage(logic.AOI_PROCESSING)
        if mine_cov < MIN_VALID_COVERAGE:
            print(
                f"  skip {day}: mine validity {mine_cov:.0%} "
                f"(processing AOI {proc_cov:.0%}; min mine {MIN_VALID_COVERAGE:.0%})",
                flush=True,
            )
            continue
        print(
            f"  keep {day}: mine validity {mine_cov:.0%} (processing AOI {proc_cov:.0%})",
            flush=True,
        )
        paths.append(path)
        days.append(_day(product.properties))

    print(
        f"  boxcar {WINDOW_RG}×{WINDOW_AZ}; min_valid_coverage={MIN_VALID_COVERAGE:.0%}",
        flush=True,
    )
    mine_stack, ref_stack = [], []
    for index in range(len(paths) - 1):
        gap = (days[index + 1] - days[index]).days
        if gap > logic.MAX_PAIR_DAYS:
            print(
                f"  skip pair {days[index]} x {days[index + 1]}, {gap} days "
                f"(max {logic.MAX_PAIR_DAYS})",
                flush=True,
            )
            continue
        print(
            f"Pair {len(mine_stack) + 1}  {paths[index].name}  x  {paths[index + 1].name}",
            flush=True,
        )
        a, ta, ca = _read_complex(paths[index], pin)
        b, tb, cb = _read_complex(paths[index + 1], pin)
        if ta != tb or ca != cb or a.shape != b.shape:
            raise RuntimeError("NISAR pair grids are not aligned — check subset downloads")
        coh = _boxcar_coherence(a, b)
        mine = _to_wgs84_pair(coh, ta, ca, logic.AOI_MINE, days[index], days[index + 1])
        ref = _to_wgs84_pair(coh, ta, ca, logic.AOI_STABLE_POINTS, days[index], days[index + 1])
        mine_stack.append(mine)
        ref_stack.append(ref)
    return mine_stack, ref_stack
