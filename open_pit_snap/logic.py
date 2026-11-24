"""Notebook SNAP/SAFE/NDAI logic ported from mine_analysis.ipynb.

Module-level names below are filled by ``open_pit_snap.runtime.bind(project)``
before any workflow function runs.
"""
from __future__ import annotations

import ee
import re
import json
import zipfile
import numpy as np
import asf_search as asf
from pathlib import Path
from scipy.ndimage import zoom, label
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from rasterio.transform import Affine
from shapely import wkt as shapely_wkt
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from datetime import datetime, timedelta
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import unary_union, transform
from asf_search.constants import INTERNAL as ASF_INTERNAL

ProductIO = GPF = HashMap = jpy = None
Integer = Double = Boolean = GeoPos = PixelPos = None

if "asf-urs" not in ASF_INTERNAL.AUTH_COOKIES:
    ASF_INTERNAL.AUTH_COOKIES = list(ASF_INTERNAL.AUTH_COOKIES) + ["asf-urs"]


def _ensure_snap():
    """Load esa_snappy on first SNAP call (keeps --help usable without SNAP)."""
    global ProductIO, GPF, HashMap, jpy, Integer, Double, Boolean, GeoPos, PixelPos
    if ProductIO is not None:
        return
    try:
        from esa_snappy import ProductIO as _ProductIO, GPF as _GPF, HashMap as _HashMap, jpy as _jpy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "esa_snappy (SNAP Python) is required for open-pit-snap. "
            "Install SNAP and esa_snappy on Windows, then pip-install this package."
        ) from exc
    ProductIO, GPF, HashMap, jpy = _ProductIO, _GPF, _HashMap, _jpy
    Integer = jpy.get_type("java.lang.Integer")
    Double = jpy.get_type("java.lang.Double")
    Boolean = jpy.get_type("java.lang.Boolean")
    GeoPos = jpy.get_type("org.esa.snap.core.datamodel.GeoPos")
    PixelPos = jpy.get_type("org.esa.snap.core.datamodel.PixelPos")

# Bound at runtime from project.yaml
START_DATE = None
END_DATE = None
STABLE_POINTS_LONGITUDE = None
STABLE_POINTS_LATITUDE = None
VOIVODESHIP = None
EARTHDATA_TOKEN = None
GEE_PROJECT_NAME = None
DOWNLOAD_DIR = None
OUTPUT_DIR = None
COHERENCE_DIR = None
POLYGON_PATH = None
AOI_MINE = None
AOI_STABLE_POINTS = None
AOI_PROCESSING = None
MINE_NAME = None
MINIMUM_SCENES = 5
MAX_PAIR_DAYS = 13
MIN_TEMPORAL_COVERAGE = 0.8
MIN_MEAN_COHERENCE = 0.8
MAX_COHERENCE_STD = 0.2
MIN_CLUSTER_PIXELS = 3
MIN_STABLE_PIXELS = 30


# --- notebook cell 6 ---
def make_polygon_buffer(path):
    print("Preparing AOI for analysis ...")
    buffer = 200
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    geom = unary_union([shape(f["geometry"]) for f in feats])

    lon, lat = geom.centroid.x, geom.centroid.y
    zone = int((lon + 180) // 6) + 1
    utm_epsg = (32600 if lat >= 0 else 32700) + zone

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    geom_utm = transform(to_utm.transform, geom)
    bbox_utm = geom_utm.buffer(buffer).envelope
    bbox_wgs = transform(to_wgs.transform, bbox_utm)

    print("  polygon bounds:", geom.bounds)
    print("  AOI bounds:", bbox_wgs.bounds)
    print("  buffer: %d m" % buffer)

    return bbox_wgs.wkt


def make_reference_window():
    print("Preparing reference window %s ...")
    buffer = 1500
    zone = int((STABLE_POINTS_LONGITUDE + 180) // 6) + 1
    utm_epsg = (32600 if STABLE_POINTS_LATITUDE >= 0 else 32700) + zone
    
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:%d" % utm_epsg, always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:%d" % utm_epsg, "EPSG:4326", always_xy=True)

    center = shapely_wkt.loads("POINT (%s %s)" % (STABLE_POINTS_LONGITUDE, STABLE_POINTS_LATITUDE))
    center_utm = transform(to_utm.transform, center)
    bbox_wgs = transform(to_wgs.transform, center_utm.buffer(buffer).envelope)

    print("  window bounds:", bbox_wgs.bounds)
    print("  buffer: %d m" % buffer)
    return bbox_wgs.wkt


# --- notebook cell 8 ---
def _scene_day(path):
    return datetime.strptime(re.search(r"(\d{8})T", Path(path).name).group(1), "%Y%m%d")


def _search_in_geo_search():
    start = datetime.combine(
        datetime.strptime(START_DATE, "%Y-%m-%d").date(),
        datetime.min.time(),
    )
    end = datetime.combine(
        datetime.strptime(END_DATE, "%Y-%m-%d").date(),
        datetime.max.time(),
    )
    assert end >= start, "END_DATE < START_DATE"

    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=[asf.PRODUCT_TYPE.SLC],
        beamMode=[asf.BEAMMODE.IW],
        polarization=[asf.POLARIZATION.VV_VH],
        intersectsWith=AOI_PROCESSING,
        start=start,
        end=end,
    )

    whole_aoi_inside = []
    for r in results:
        scene = shape(r.geojson()["geometry"])
        if scene.covers(shapely_wkt.loads(AOI_PROCESSING)):
            whole_aoi_inside.append(r)
    return sorted(whole_aoi_inside, key=lambda r: r.properties.get("startTime", ""))


def _prepare_scene_stack(results):
    assert results, "  No scene in date %s .. %s" % (START_DATE, END_DATE)
    first = results[0]
    rel_orbit = first.properties.get("pathNumber")
    flight_dir = first.properties.get("flightDirection")
    stack = []
    seen_dates = set()

    winter_days = []
    for r in results:
        p = r.properties
        if p.get("pathNumber") != rel_orbit:
            continue
        if p.get("flightDirection") != flight_dir:
            continue
        day = (p.get("startTime") or "")[:10]
        if not day or day in seen_dates:
            continue
        if int(day[5:7]) in (12, 1, 2):
            winter_days.append(day)
            continue
        seen_dates.add(day)
        stack.append(r)
    
    if winter_days:
        print("  rejected %d winter scenes: %s" % (len(winter_days), ", ".join(winter_days)))
    return stack, rel_orbit, flight_dir


def find_sentinel_1_scenes():
    print("Searching for Sentinel-1 scenes ...")

    raw_results = _search_in_geo_search()
    scene_stack, rel_orbit, flight_dir = _prepare_scene_stack(raw_results)

    print("  scenes found: %d" % len(scene_stack))
    print("  date range: %s - %s" % (START_DATE, END_DATE))
    print("  relative orbit = %s;  flight direction = %s" % (rel_orbit, flight_dir))

    for i, r in enumerate(scene_stack, 1):
        p = r.properties
        name = p.get("fileName")
        day = (p.get("startTime"))[:10]
        print("  %2d. %s  %s" % (i, day, name))

    if len(scene_stack) < MINIMUM_SCENES:
        raise RuntimeError(
            "Need at least %d scenes for analysis — got %d"
            % (MINIMUM_SCENES, len(scene_stack))
        )
    return scene_stack


# --- notebook cell 10 ---
def _zip_is_complete(path, product):
    expected = product.properties.get("bytes")
    if path.stat().st_size != int(expected):
        return False
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any(name.endswith("manifest.safe") for name in archive.namelist())


def start_asf_session():
    session = asf.ASFSession()
    session.auth_cookie_names = list(
        dict.fromkeys(list(session.auth_cookie_names) + ["asf-urs"])
    )
    session.auth_with_token(EARTHDATA_TOKEN)
    print("ASF Data Search Vertex session opened ...")
    return session


def close_asf_session(session):
    session.close()
    print("ASF Data Search Vertex session closed.")


def initialize_gee():
    ee.Initialize(project=GEE_PROJECT_NAME)


def download_images(product, session):
    image = product.properties.get("fileName")
    dest = Path(DOWNLOAD_DIR) / image

    print("Download: %s" % image)
    if dest.exists() and _zip_is_complete(dest, product):
        print("  File is already downloaded: %s (%.2f GB)" % (dest.resolve(), dest.stat().st_size / 1e9))
        return dest
    if dest.exists():
        print("  Incomplete file, downloading again: %s (%.2f GB)" % (dest.name, dest.stat().st_size / 1e9))
        _release_zip_sources(dest)
        dest.unlink()

    print("Downloading: %s ..." % image)
    product.download(path=str(Path(DOWNLOAD_DIR)), session=session)

    if not _zip_is_complete(dest, product):
        _release_zip_sources(dest)
        dest.unlink()
        raise RuntimeError("  Downloaded file is incomplete: %s" % dest.name)
        
    print("Downloaded: %s (%.2f GB)" % (dest.name, dest.stat().st_size / 1e9))
    return dest


# --- notebook cell 12 ---
def open_slc(zip_path):
    _ensure_snap()
    product = ProductIO.readProduct(str(zip_path))
    print("Opened: ", product.getName())
    return product


# --- notebook cell 14 ---
def apply_orbit_file(product):
    _ensure_snap()
    print("Apply Orbit File ...")
    
    params = HashMap()
    params.put("orbitType", "Sentinel Precise (Auto Download)")
    params.put("polyDegree", Integer(3))
    params.put("continueOnFail", Boolean.FALSE)
    
    orbit = GPF.createProduct("Apply-Orbit-File", params, product)
    return orbit


# --- notebook cell 16 ---
def _swath_to_rectangle(product, band_name):
    band = product.getBand(band_name)
    gc = band.getGeoCoding()
    w, h = band.getRasterWidth(), band.getRasterHeight()
    corners = []

    for x, y in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)):
        gp = GeoPos()
        gc.getGeoPos(PixelPos(float(x), float(y)), gp)
        corners.append((gp.lon, gp.lat))
    return Polygon(corners)


def choose_subswaths_for_aoi(product):
    swaths = []

    for swath in ("IW1", "IW2", "IW3"):
        rectangle = _swath_to_rectangle(product, f"Intensity_{swath}_VV")
        intersection = rectangle.intersection(shapely_wkt.loads(AOI_PROCESSING))
        area = intersection.area if not intersection.is_empty else 0.0
        if area > 0:
            swaths.append(swath)

    print("  subswaths: %s" % ", ".join(swaths))
    return swaths


def topsar_split(product, subswath):
    print("S-1 TOPS Split ...")

    params = HashMap()
    params.put("subswath", subswath)
    params.put("selectedPolarisations", "VV")
    params.put("wktAoi", AOI_PROCESSING)
    
    split = GPF.createProduct("TOPSAR-Split", params, product)
    return split


# --- notebook cell 18 ---
def back_geocoding(master, slave):
    print("S-1 Back-Geocoding: Copernicus 30m, bilinear ...")

    params = HashMap()
    params.put("demName", "Copernicus 30m Global DEM")
    params.put("demResamplingMethod", "BILINEAR_INTERPOLATION")
    params.put("resamplingType", "BILINEAR_INTERPOLATION")
    params.put("maskOutAreaWithoutElevation", Boolean.TRUE)
    params.put("outputRangeAzimuthOffset", Boolean.FALSE)
    params.put("outputDerampDemodPhase", Boolean.FALSE)
    params.put("disableReramp", Boolean.FALSE)
    Product = jpy.get_type("org.esa.snap.core.datamodel.Product")
    sources = jpy.array(Product, 2)
    sources[0] = master
    sources[1] = slave

    geocoding = GPF.createProduct("Back-Geocoding", params, sources)
    return geocoding


# --- notebook cell 20 ---
def coherence_estimation(raster):
    print("Coherence Estimation: Copernicus 30m, topographic phase, window 10x3 ...")

    params = HashMap()
    params.put("subtractFlatEarthPhase", Boolean.TRUE)
    params.put("srpPolynomialDegree", Integer(5))
    params.put("srpNumberPoints", Integer(501))
    params.put("orbitDegree", Integer(3))
    params.put("squarePixel", Boolean.FALSE)
    params.put("cohWinRg", Integer(10))
    params.put("cohWinAz", Integer(3))
    params.put("subtractTopographicPhase", Boolean.TRUE)
    params.put("demName", "Copernicus 30m Global DEM")
    params.put("demResamplingMethod", "BILINEAR_INTERPOLATION")

    coherence = GPF.createProduct("Coherence", params, raster)
    return coherence


# --- notebook cell 22 ---
def topsar_deburst(raster):
    print("S-1 TOPS Deburst ...")

    params = HashMap()
    params.put("selectedPolarisations", "VV")
    
    deburst = GPF.createProduct("TOPSAR-Deburst", params, raster)
    return deburst


# --- notebook cell 24 ---
def topsar_merge(raster):
    print("S-1 TOPS Merge ...")

    params = HashMap()
    params.put("selectedPolarisations", "VV")
    Product = jpy.get_type("org.esa.snap.core.datamodel.Product")
    sources = jpy.array(Product, len(raster))
    for i, product in enumerate(raster):
        sources[i] = product

    merged = GPF.createProduct("TOPSAR-Merge", params, sources)
    return merged


# --- notebook cell 26 ---
def multilook(raster):
    print("Multilook: 6 range x 1 azimuth ...")

    params = HashMap()
    params.put("nRgLooks", Integer(6))
    params.put("nAzLooks", Integer(1))
    params.put("outputIntensity", Boolean.FALSE)
    params.put("grSquarePixel", Boolean.FALSE)

    looked = GPF.createProduct("Multilook", params, raster)
    return looked


# --- notebook cell 28 ---
def terrain_correction(raster):
    print("Range-Doppler Terrain Correction: Copernicus 30m, bilinear ...")

    WGS84 = (
        'GEOGCS["WGS84(DD)",'
        'DATUM["WGS84",SPHEROID["WGS84",6378137.0,298.257223563]],'
        'PRIMEM["Greenwich",0.0],'
        'UNIT["degree",0.017453292519943295],'
        'AXIS["Geodetic longitude",EAST],'
        'AXIS["Geodetic latitude",NORTH],'
        'AUTHORITY["EPSG","4326"]]'
    )

    params = HashMap()
    params.put("demName", "Copernicus 30m Global DEM")
    params.put("demResamplingMethod", "BILINEAR_INTERPOLATION")
    params.put("imgResamplingMethod", "BILINEAR_INTERPOLATION")
    params.put("mapProjection", WGS84)
    params.put("pixelSpacingInMeter", Double(13.92475))
    params.put("nodataValueAtSea", Boolean.TRUE)
    params.put("saveSelectedSourceBand", Boolean.TRUE)
    params.put("saveDEM", Boolean.FALSE)
    params.put("saveLatLon", Boolean.FALSE)
    params.put("saveLocalIncidenceAngle", Boolean.FALSE)
    params.put("saveLayoverShadowMask", Boolean.TRUE)
    params.put("outputComplex", Boolean.FALSE)
    params.put("alignToStandardGrid", Boolean.TRUE)
    params.put("applyRadiometricNormalization", Boolean.FALSE)

    terrain_correction = GPF.createProduct("Terrain-Correction", params, raster)
    return terrain_correction


# --- notebook cell 30 ---
def subset(raster, region):
    print("Subset ...")

    params = HashMap()
    params.put("geoRegion", region)
    params.put("copyMetadata", Boolean.TRUE)
    params.put("fullSwath", Boolean.FALSE)
    
    subset = GPF.createProduct("Subset", params, raster)
    return subset


# --- notebook cell 32 ---
def _band_name(raster, key):
    for name in raster.getBandNames():
        if key(str(name).lower()):
            return str(name)
    return None


def _read_pixels(band):
    width = band.getRasterWidth()
    height = band.getRasterHeight()
    arr = np.empty(width * height, dtype=np.float32)
    band.readPixels(0, 0, width, height, arr)
    return arr.reshape(height, width)


def _scene_grid(raster):
    gc = raster.getSceneGeoCoding()
    width = raster.getSceneRasterWidth()
    height = raster.getSceneRasterHeight()

    def geo(x, y):
        gp = GeoPos()
        gc.getGeoPos(PixelPos(float(x), float(y)), gp)
        return float(gp.lon), float(gp.lat)

    lon00, lat00 = geo(0, 0)
    lon10, _ = geo(1, 0)
    _, lat01 = geo(0, 1)
    dx = lon10 - lon00
    dy = lat01 - lat00
    grid = Affine(dx, 0.0, lon00 - dx / 2.0, 0.0, dy, lat00 - dy / 2.0)
    left = grid.c
    top = grid.f
    extent = (left, left + dx * width, top + dy * height, top)
    return grid, extent


def raster_to_array(raster, master_zip, slave_zip, label):
    print("Convert raster %s to array ..." % label)
    master_day = _scene_day(master_zip)
    slave_day = _scene_day(slave_zip)

    coherence_name = _band_name(raster, lambda name: name.startswith("coh"))
    coh_pixels = _read_pixels(raster.getBand(coherence_name))
    coherence = np.where((coh_pixels < 0) | (coh_pixels > 1), np.nan, coh_pixels).astype(np.float32, copy=False)

    shadow_name = _band_name(raster, lambda name: "layover" in name or "shadow" in name)
    shadow = _read_pixels(raster.getBand(shadow_name)) if shadow_name is not None else None

    grid, extent = _scene_grid(raster)
    return {
        "master_day": master_day,
        "slave_day": slave_day,
        "coherence": coherence,
        "shadow": shadow,
        "transform": grid,
        "extent": extent,
    }


# --- notebook cell 34 ---
def _polygons_from_gee(collection):
    features = []
    token = None
    while True:
        request = {"expression": collection, "pageSize": 1000}
        if token:
            request["pageToken"] = token
        page = ee.data.computeFeatures(request)
        features.extend(page["features"])
        token = page.get("nextPageToken")
        if not token:
            break
    polygons = [shape(feature["geometry"]) for feature in features]
    return polygons


def _rasterize_polygon(polygons, shape_hw, grid_transform):
    mask =  rasterize(
        [(geom, 1) for geom in polygons],
        out_shape=shape_hw,
        transform=grid_transform,
        fill=0,
        dtype="uint8",
    )
    return mask


def _cluster_size_mask(mask, min_pixels, structure):
    labeled, n = label(mask, structure=structure)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labeled]


def _show_stable_points(mean, stable):
    fig, ax = plt.subplots(figsize=(8, 8))
    show = np.ma.masked_invalid(mean)
    ax.imshow(show, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros(mean.shape + (4,), dtype=np.float32)
    overlay[stable] = (0.88, 0.04, 0.0, 0.85)
    ax.imshow(overlay)
    ax.set_axis_off()
    fig.tight_layout()
    plt.close(fig)


def buildings_in_reference_window(references):
    print("Stable points ...")
    base = references[0]
    initialize_gee()

    aoi = ee.Geometry(shapely_wkt.loads(AOI_STABLE_POINTS).__geo_interface__)
    shape_hw = base["coherence"].shape
    grid_transform = base["transform"]

    folder = "projects/%s/assets/%s" % (GEE_PROJECT_NAME, VOIVODESHIP)
    buildings = _polygons_from_gee(ee.FeatureCollection("%s/buildings" % folder).filterBounds(aoi))
    if not buildings:
        raise RuntimeError("No building in the reference window")
    return _rasterize_polygon(buildings, shape_hw, grid_transform).astype(bool)


def pixels_in_enough_pairs(references, building_mask):
    stack = np.stack([ref["coherence"] for ref in references], axis=0)
    count = np.sum(np.isfinite(stack), axis=0)
    enough = count >= int(MIN_TEMPORAL_COVERAGE * stack.shape[0])
    covered = building_mask & enough
    print("  building pixels in the reference window: %d" % int(covered.sum()))
    return covered


def stable_pixels(references, covered):
    stack = np.stack([ref["coherence"] for ref in references], axis=0)
    mean = np.nanmean(np.where(covered, stack, np.nan), axis=0)
    std = np.nanstd(np.where(covered, stack, np.nan), axis=0)
    # One pair → temporal std is undefined; treat as 0 so the mean cut still applies.
    if stack.shape[0] < 2:
        std = np.where(covered, 0.0, np.nan)
    strict = covered & (mean > MIN_MEAN_COHERENCE) & (std < MAX_COHERENCE_STD)
    print(
        "  building pixels with mean > %.2f and std < %.2f: %d"
        % (MIN_MEAN_COHERENCE, MAX_COHERENCE_STD, int(strict.sum()))
    )
    return strict, mean, std


def compact_stable_pixels(strict):
    stable = _cluster_size_mask(strict, MIN_CLUSTER_PIXELS, np.ones((3, 3), dtype=int))
    print("  after removing isolated pixels, stable pixels accepted: %d" % int(stable.sum()))
    if int(stable.sum()) < MIN_STABLE_PIXELS:
        raise RuntimeError(
            "Fewer than %d compact building pixels. The analysis cannot continue."
            % MIN_STABLE_PIXELS
        )
    return stable


def gamma_stable_per_pair(references, stable, mean, std):
    stack = np.stack([ref["coherence"] for ref in references], axis=0)

    print("  stable coherence mean: %.3f" % float(np.nanmean(mean[stable])))
    print("  stable coherence std: %.3f" % float(np.nanmean(std[stable])))

    pairs = []
    gammas = []
    for ref, image in zip(references, stack):
        values = image[stable]
        values = values[np.isfinite(values)]
        gamma = float(np.mean(values)) if values.size else float("nan")
        gammas.append(gamma)
        pairs.append({
            "master_day": ref["master_day"],
            "slave_day": ref["slave_day"],
            "gamma_stable": None if np.isnan(gamma) else round(gamma, 4),
        })

    center = float(np.nanmedian(gammas))
    kept = []
    for pair, gamma in zip(pairs, gammas):
        if not np.isnan(gamma) and gamma >= center - 0.10:
            kept.append(pair)
        else:
            print(
                "  reject pair %s - %s, gamma_stable %.3f"
                % (pair["master_day"].strftime("%Y.%m.%d"), pair["slave_day"].strftime("%Y.%m.%d"), gamma)
            )

    print("  pairs kept: %d" % len(kept))
    if not kept:
        raise RuntimeError("Every pair was rejected. The analysis cannot continue.")

    _show_stable_points(mean, stable)
    return kept


# --- notebook cell 36 ---
def _ten_meter_grid(extent):
    minx, maxx, miny, maxy = extent
    lon = 0.5 * (minx + maxx)
    lat = 0.5 * (miny + maxy)
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True)
    corners = ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy))
    east, north = zip(*(to_utm.transform(x, y) for x, y in corners))
    left = float(np.floor(min(east) / 10.0) * 10.0)
    right = float(np.ceil(max(east) / 10.0) * 10.0)
    bottom = float(np.floor(min(north) / 10.0) * 10.0)
    top = float(np.ceil(max(north) / 10.0) * 10.0)
    width = int(round((right - left) / 10.0))
    height = int(round((top - bottom) / 10.0))
    transform = Affine(10.0, 0.0, left, 0.0, -10.0, top)
    return epsg, transform, height, width


def _bands_on_10m(image, bands, extent):
    epsg, transform, height, width = _ten_meter_grid(extent)
    struct = ee.data.computePixels({
        "expression": image.unmask(-9999),
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": list(bands),
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": 10,
                "shearX": 0,
                "translateX": transform.c,
                "shearY": 0,
                "scaleY": -10,
                "translateY": transform.f,
            },
            "crsCode": "EPSG:%d" % epsg,
        },
    })
    arrays = {}
    for band in bands:
        values = np.asarray(struct[band], dtype=np.float32)
        values[values < -1.5] = np.nan
        arrays[band] = values
    return arrays, transform, "EPSG:%d" % epsg


def _reproject_boolean(mask, src_transform, src_crs, shape_hw, grid_transform):
    dest = np.zeros(shape_hw, dtype=np.uint8)
    reproject(
        source=np.asarray(mask, dtype=np.uint8),
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )
    return dest.astype(bool)


def _extend_date_range_by_month(day, months):
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    lengths = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return day.replace(year=year, month=month, day=min(day.day, lengths[month - 1]))


def _image_label(image):
    meta = ee.Dictionary({
        "id": image.get("system:index"),
        "date": ee.Date(image.get("system:time_start")).format("YYYY.MM.dd"),
    }).getInfo()
    return "%s %s" % (meta["date"], meta["id"])


def _contains_whole_aoi(image, aoi):
    contains_aoi = bool(image.geometry().contains(aoi, ee.ErrorMargin(1)).getInfo())
    return contains_aoi


def _check_clouds(image, aoi):
    scl = image.select("SCL")
    cloud = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).rename("cloud"))
    clouds_fraction = cloud.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e8,
    ).get("cloud").getInfo()
    return clouds_fraction


def cloud_mask(image, extent, shape_hw, grid_transform):
    print("  cloud mask from SCL classes 3, 8, 9 and 10 ...")
    scene_class = image.select(["SCL"])
    arrays, src_transform, src_crs = _bands_on_10m(scene_class, ["SCL"], extent)
    scl = arrays["SCL"]
    mask = np.isfinite(scl) & ((scl == 3) | (scl == 8) | (scl == 9) | (scl == 10))
    
    cloud = _reproject_boolean(mask, src_transform, src_crs, shape_hw, grid_transform)
    return cloud


def ndvi_swir_mask(image, extent, shape_hw, grid_transform, cloud):
    print("  vegetation mask from NDVI >= 0.40 and SWIR < 0.26, outside the cloud mask ...")
    ndvi = image.select(["B8", "B4"]).divide(10000).normalizedDifference(["B8", "B4"]).rename("NDVI")
    swir = image.select(["B11"]).divide(10000).rename("SWIR")
    arrays, src_transform, src_crs = _bands_on_10m(ndvi.addBands(swir), ["NDVI", "SWIR"], extent)
    
    mask = (
        np.isfinite(arrays["NDVI"]) & (arrays["NDVI"] >= 0.42)
        & np.isfinite(arrays["SWIR"]) & (arrays["SWIR"] < 0.26)
    )
    vegetation = _reproject_boolean(mask, src_transform, src_crs, shape_hw, grid_transform)
    vegetation = vegetation & ~cloud

    if not vegetation.any():
        print("    no NDVI and SWIR vegetation in the mine area")
    return vegetation

def water_mask(image, extent, shape_hw, grid_transform, vegetation, cloud):
    print("  water mask from WIW, dark water or NDWI >= 0, outside the vegetation mask and the cloud mask ...")
    reflectance = image.select(["B2", "B3", "B4", "B8", "B11", "B12"]).divide(10000)
    arrays, src_transform, src_crs = _bands_on_10m(
        reflectance,
        ["B2", "B3", "B4", "B8", "B11", "B12"],
        extent,
    )

    blue = arrays["B2"]
    green = arrays["B3"]
    red = arrays["B4"]
    nir = arrays["B8"]
    swir1 = arrays["B11"]
    swir2 = arrays["B12"]

    brightness = (blue + green + red) / 3.0
    ndwi = (green - nir) / (green + nir)
    finite = (
        np.isfinite(brightness) & np.isfinite(ndwi) & np.isfinite(nir) & np.isfinite(swir1)
        & np.isfinite(swir2) & np.isfinite(red)
    )
    dark_water = (brightness < 0.08) & (swir1 < 0.16)
    wiw = (nir <= 0.18) & (swir2 <= 0.11)
    ndwi = (ndwi >= 0)

    mask = finite & (dark_water | wiw | ndwi)
    water = _reproject_boolean(mask, src_transform, src_crs, shape_hw, grid_transform)
    water = water & ~vegetation & ~cloud

    if not water.any():
        print("    no water on the image")
    return water


def find_sentinel2_images(start, end, first_only=False):
    print("  searching for Sentinel-2 images ...")
    print("    date range: %s - %s" % (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

    aoi = ee.Geometry(shapely_wkt.loads(AOI_MINE).__geo_interface__)

    start_day = start.date() if isinstance(start, datetime) else start
    end_day = end.date() if isinstance(end, datetime) else end
    start = datetime.combine(start_day, datetime.min.time())
    end = datetime.combine(end_day, datetime.max.time())

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    total = col.size().getInfo()
    if not total:
        return []

    listed = col.toList(total)
    image_accepted = []
    for image in range(total):
        image = ee.Image(listed.get(image))

        if not _contains_whole_aoi(image, aoi):
            continue

        clouds = _check_clouds(image, aoi)
        if clouds is None or clouds > 0:
            continue

        if first_only:
            return image
        
        image_accepted.append(image)

    return image_accepted


# --- notebook cell 38 ---
def _treshold_rules(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, [0.005, 0.995])
    clipped = np.clip(values, low, high)
    if np.std(clipped) < 1e-6:
        return float(np.median(clipped)), "median, no spread"
    if clipped.size > 200000:
        clipped = np.random.default_rng(0).choice(clipped, 200000, replace=False)

    density_grid = np.linspace(float(low), float(high), 512)
    density = gaussian_kde(clipped)(density_grid)
    slope = np.sign(np.diff(density))
    valleys = np.where((slope[:-1] < 0) & (slope[1:] > 0))[0] + 1
    span = float(high - low)
    interior = [
        index for index in valleys
        if low + 0.1 * span < density_grid[index] < high - 0.1 * span
    ]
    if len(interior):
        best = min(interior, key=lambda index: density[index])
        return float(density_grid[best]), "density minimum"

    hist, edges = np.histogram(clipped, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(hist).astype(np.float64)
    weight_high = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    sum_low = np.cumsum(hist * centers)
    sum_high = np.cumsum((hist * centers)[::-1])[::-1]
    mean_low = sum_low / np.maximum(weight_low, 1)
    mean_high = sum_high / np.maximum(weight_high, 1)
    between = weight_low[:-1] * weight_high[1:] * (mean_low[:-1] - mean_high[1:]) ** 2
    return float(centers[int(np.argmax(between))]), "otsu, no interior minimum"


def coherence_mine_grid(stack_coherence_mine, gamma_stable_for_raster):
    by_pair = {
        (raster["master_day"], raster["slave_day"]): raster
        for raster in stack_coherence_mine
    }
    first = gamma_stable_for_raster[0]
    mine_raster = by_pair[(first["master_day"], first["slave_day"])]
    
    extent = mine_raster["extent"]
    grid_transform = mine_raster["transform"]
    shape_hw = mine_raster["coherence"].shape

    mine = shape(json.loads(POLYGON_PATH.read_text(encoding="utf-8"))["features"][0]["geometry"])
    mine_mask = _rasterize_polygon([mine], shape_hw, grid_transform).astype(bool)
    return by_pair, extent, grid_transform, shape_hw, mine, mine_mask


def shadow_mask(record, shape_hw):
    shadow = record["shadow"]
    if shadow is None:
        if not getattr(shadow_mask, "announced", False):
            print("  shadow mask is missing, radar shadow is not removed.")
            shadow_mask.announced = True
        return np.zeros(shape_hw, dtype=bool)
    return np.isfinite(shadow) & (shadow > 0)


def ndai_index(coherence, gamma_stable):
    numerator = gamma_stable - coherence
    denominator = gamma_stable + coherence
    ndai = np.full(coherence.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(coherence) & (np.abs(denominator) > 1e-6)
    ndai[valid] = numerator[valid] / denominator[valid]
    return ndai


def ndai_samples_inside_mine(ndai, mine_mask, water, vegetation, shadow, cloud):
    ndai_samples = ndai[mine_mask & ~water & ~vegetation & ~shadow & ~cloud & np.isfinite(ndai)]
    return ndai_samples


def ndai_threshold(samples):
    if not samples:
        raise RuntimeError("No NDAI samples inside the mine")
    pooled = np.concatenate(samples)
    threshold, rule = _treshold_rules(pooled)
    print("  NDAI threshold: %.4f, rule used: %s)" % (threshold, rule))
    return threshold


# --- notebook cell 40 ---
def changed_area_inside_polygon(change, mine_mask, grid_transform):
    mine = np.asarray(mine_mask, dtype=bool)
    inside = np.asarray(change, dtype=bool) & mine
    count = int(inside.sum())
    mine_count = int(mine.sum())
    percent = 100.0 * count / mine_count if mine_count else 0.0
    rows = np.flatnonzero(np.asarray(mine_mask).any(axis=1))
    row = 0.5 * (rows[0] + rows[-1]) if rows.size else mine_mask.shape[0] / 2.0
    lat = grid_transform.f + grid_transform.e * row
    lat_rad = np.deg2rad(lat)
    meters_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    meters_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    pixel_area = abs(float(grid_transform.a)) * meters_lon * abs(float(grid_transform.e)) * meters_lat
    area = count * pixel_area
    print("  area altered: %.2f ha (%.1f%% of the mine)" % (area / 10000.0, percent))
    return area, percent


def activity_map(array, polygon, master_day, slave_day, extent, area, percent):
    ...


def activity_map_with_masks(array, polygon, master_day, slave_day, extent, area, percent, water, vegetation, shadow, cloud):
    ...


def activity_map_with_sentinel2_references(array, polygon, master_day, slave_day, extent, area, percent):
    ...


def activity_mask(pair, threshold):
    ndai_array = pair["ndai"]
    change = np.isfinite(ndai_array) & (ndai_array > threshold) & ~pair["water"] & ~pair["vegetation"] & ~pair["shadow"] & ~pair["cloud"]
    return _cluster_size_mask(change, 3, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))


# --- notebook cell 42 ---
def _visualization_info(master_day, slave_day, area, percent):
    dates = "%s - %s" % (master_day.strftime("%d.%m.%Y"), slave_day.strftime("%d.%m.%Y"))
    area_changed = "area altered: %.2f ha (%.1f%% of the mine)" % (area / 10000.0, percent)
    mine_name = MINE_NAME.replace("_", " ").title()
    return dates, area_changed, mine_name


def _draw_outline(ax, arr, geom, extent):
    if isinstance(geom, str):
        geom = shapely_wkt.loads(geom)
    if geom.geom_type == "Polygon":
        parts = [geom]
    else:
        parts = [part for part in geom.geoms if part.geom_type == "Polygon"]

    minx, maxx, miny, maxy = extent
    arr = np.asarray(arr)
    height, width = arr.shape[:2]

    for part in parts:
        rings = [part.exterior, *part.interiors]
        for ring in rings:
            xs, ys = ring.xy
            cols = (np.asarray(xs) - minx) / (maxx - minx) * width - 0.5
            rows = (maxy - np.asarray(ys)) / (maxy - miny) * height - 0.5
            ax.plot(cols, rows, color="#e10600", linewidth=1.4)


def _activity_rgb(array, water, vegetation, shadow, cloud=None):
    gray = np.clip(np.asarray(array, dtype=np.float32), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if vegetation is not None:
        rgb[np.asarray(vegetation, dtype=bool)] = (50 / 255, 205 / 255, 50 / 255)
    if water is not None:
        rgb[np.asarray(water, dtype=bool)] = (135 / 255, 206 / 255, 250 / 255)
    if cloud is not None:
        rgb[np.asarray(cloud, dtype=bool)] = (168 / 255, 168 / 255, 168 / 255)
    if shadow is not None:
        rgb[np.asarray(shadow, dtype=bool)] = (147 / 255, 112 / 255, 219 / 255)
    return rgb


def _draw_activity_map(ax, legend_ax, array, polygon, extent, water=None, vegetation=None, shadow=None, cloud=None):
    ax.imshow(_activity_rgb(array, water, vegetation, shadow, cloud), interpolation="nearest")
    _draw_outline(ax, array, polygon, extent)
    ax.set_axis_off()
    legend_ax.set_axis_off()
    handles = [
        Patch(facecolor="white", edgecolor="black", label="surface change"),
        Patch(facecolor="black", edgecolor="black", label="no change"),
    ]
    if vegetation is not None:
        handles.append(Patch(facecolor=(50 / 255, 205 / 255, 50 / 255), edgecolor="black", label="vegetation"))
    if water is not None:
        handles.append(Patch(facecolor=(135 / 255, 206 / 255, 250 / 255), edgecolor="black", label="water"))
    if cloud is not None:
        handles.append(Patch(facecolor=(168 / 255, 168 / 255, 168 / 255), edgecolor="black", label="cloud"))
    if shadow is not None:
        handles.append(Patch(facecolor=(147 / 255, 112 / 255, 219 / 255), edgecolor="black", label="shadow"))
    handles.append(Line2D([0], [0], color="#e10600", linewidth=2, label="mine area"))
    legend = legend_ax.legend(
        handles=handles,
        title="Legend",
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        borderaxespad=0,
        borderpad=0.8,
        ncol=3,
        columnspacing=1.6,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=1,
        fontsize=10.5,
        title_fontsize=12,
    )
    legend._legend_box.sep = 1.0 * 10.5


def _render_activity_map(array, polygon, master_day, slave_day, extent, area, percent, water, vegetation, shadow, file_suffix, legend_h, cloud=None):
    dates, area_changed, mine_name = _visualization_info(master_day, slave_day, area, percent)

    title_size = 16
    mine_size = 13
    area_size = 10
    text_gap = 0.34 - title_size / 72.0
    map_w, gap = 7.0, 0.24
    margin_v = 0.27
    margin_h = 0.27
    title_top = margin_v
    mine_top = title_top + title_size / 72.0 + text_gap
    area_top = mine_top + mine_size / 72.0 + text_gap
    title_h = (area_top - margin_v) + area_size / 72.0 + 0.08
    arr_h, arr_w = np.asarray(array).shape[:2]
    map_h = map_w * arr_h / arr_w
    fig_w = 2 * margin_h + map_w
    fig_h = 2 * margin_v + title_h + gap + map_h + gap + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    legend_ax = fig.add_axes([margin_h / fig_w, margin_v / fig_h, map_w / fig_w, legend_h / fig_h])
    ax = fig.add_axes(
        [margin_h / fig_w, (margin_v + legend_h + gap) / fig_h, map_w / fig_w, map_h / fig_h]
    )

    fig.text(
        0.5, 1 - title_top / fig_h, "Activity Map",
        fontsize=title_size, fontweight="bold", ha="center", va="top",
    )
    fig.text(
        0.5, 1 - mine_top / fig_h, "%s  |  %s" % (mine_name, dates),
        fontsize=mine_size, color="black", ha="center", va="top",
    )
    fig.text(
        0.5, 1 - area_top / fig_h, area_changed,
        fontsize=area_size, color="black", ha="center", va="top",
    )
    _draw_activity_map(ax, legend_ax, array, polygon, extent, water=water, vegetation=vegetation, shadow=shadow, cloud=cloud)

    out = OUTPUT_DIR / ("%s-%s_%s.png" % (master_day.strftime("%Y.%m.%d"), slave_day.strftime("%Y.%m.%d"), file_suffix))
    fig.savefig(out, facecolor="white")
    print("Saved map: %s" % out)
    plt.close(fig)
    return out


def _first_sentinel2_image(anchor, direction):
    first_only=True

    if direction < 0:
        start = _extend_date_range_by_month(anchor, -1)
        end = anchor - timedelta(days=1)
        first_only = False
    else:
        start = anchor + timedelta(days=1)
        end = _extend_date_range_by_month(anchor, 1)

    images = find_sentinel2_images(start, end, first_only)

    if not images:
        return None
    image = images[-1] if direction < 0 else images

    label = _image_label(image)
    print("    Sentinel-2 image scelected: %s" % label)
    date_text, _scene_id = label.split(" ", 1)
    date = datetime.strptime(date_text, "%Y.%m.%d")
    return image, date


def _sentinel_rgb(image, extent, shape_hw):
    rgb = image.select(["B4", "B3", "B2"])
    minx, maxx, miny, maxy = extent
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:4326", geodesic=False)
    import geemap
    arr = geemap.ee_to_numpy(rgb, region=region, scale=10, bands=["B4", "B3", "B2"])
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    arr = np.clip(arr[:, :, :3] / 3000.0, 0, 1)

    zy = shape_hw[0] / arr.shape[0]
    zx = shape_hw[1] / arr.shape[1]
    out = zoom(arr, (zy, zx, 1), order=1)
    fixed = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.float32)
    h = min(shape_hw[0], out.shape[0])
    w = min(shape_hw[1], out.shape[1])
    fixed[:h, :w] = out[:h, :w]
    return np.clip(fixed, 0, 1)


def activity_map_with_masks(array, polygon, master_day, slave_day, extent, area, percent, water, vegetation, shadow, cloud):
    return _render_activity_map(
        array, polygon, master_day, slave_day, extent, area, percent,
        water, vegetation, shadow, "activity_map_with_mask", 1.35, cloud
    )


def activity_map(array, polygon, master_day, slave_day, extent, area, percent):
    return _render_activity_map(
        array, polygon, master_day, slave_day, extent, area, percent,
        None, None, None, "activity_map", 0.67
    )


def activity_map_with_sentinel2_references(array, polygon, master_day, slave_day, extent, area, percent):
    dates, area_changed, mine_name = _visualization_info(master_day, slave_day, area, percent)
    shape_hw = np.asarray(array).shape[:2]

    panels = []
    for hit in (_first_sentinel2_image(master_day, -1), _first_sentinel2_image(slave_day, 1)):
        if hit is None:
            panels.append(None)
            continue
        image, date = hit
        rgb = _sentinel_rgb(image, extent, shape_hw)
        panels.append((rgb, date))

    title_size = 16
    mine_size = 16
    optical_size = 13
    activity_size = 13
    area_size = 10
    text_gap = 0.12
    pad_above_map = 0.10
    row_gap = 0.28
    col_gap = 0.40
    legend_h = 0.67
    margin_v, margin_h = 0.27, 0.27
    optical_caption_h = optical_size / 72.0 + 0.10

    arr_h, arr_w = shape_hw
    sar_w = 7.06
    sar_h = sar_w * arr_h / arr_w
    activity_caption_h = pad_above_map + area_size / 72.0 + text_gap + activity_size / 72.0
    left_h = activity_caption_h + sar_h + row_gap + legend_h
    s2_h = (left_h - 2 * optical_caption_h - row_gap) / 2
    s2_w = s2_h * arr_w / arr_h

    title_h = title_size / 72.0 + text_gap + mine_size / 72.0 + 0.16
    fig_w = 2 * margin_h + sar_w + col_gap + s2_w
    fig_h = 2 * margin_v + title_h + left_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    y_legend = margin_v
    y_sar = y_legend + legend_h + row_gap
    y_image2 = y_legend
    y_image1 = y_legend + s2_h + optical_caption_h + row_gap
    right_x = margin_h + sar_w + col_gap

    legend_ax = fig.add_axes([margin_h / fig_w, y_legend / fig_h, sar_w / fig_w, legend_h / fig_h])
    sar_ax = fig.add_axes([margin_h / fig_w, y_sar / fig_h, sar_w / fig_w, sar_h / fig_h])
    upper_ax = fig.add_axes([right_x / fig_w, y_image1 / fig_h, s2_w / fig_w, s2_h / fig_h])
    lower_ax = fig.add_axes([right_x / fig_w, y_image2 / fig_h, s2_w / fig_w, s2_h / fig_h])

    mine_top = margin_v + title_size / 72.0 + text_gap
    fig.text(
        0.5, 1 - margin_v / fig_h, "Activity Map and Optical Images Comparison",
        fontsize=title_size, fontweight="bold", ha="center", va="top",
    )
    fig.text(
        0.5, 1 - mine_top / fig_h, mine_name,
        fontsize=mine_size, color="black", ha="center", va="top",
    )

    for ax, panel in ((upper_ax, panels[0]), (lower_ax, panels[1])):
        if panel is None:
            ax.imshow(np.full((shape_hw[0], shape_hw[1], 3), 0.94, dtype=np.float32), interpolation="nearest")
            ax.text(
                shape_hw[1] / 2, shape_hw[0] / 2,
                "No image found\nwithin one month",
                ha="center", va="center", fontsize=11, color="#333333",
            )
            caption = "Optical Image"
        else:
            rgb, date = panel
            ax.imshow(rgb, interpolation="nearest", aspect="auto")
            _draw_outline(ax, array, polygon, extent)
            caption = "Optical Image  |  %s" % date.strftime("%d.%m.%Y")
        ax.set_axis_off()
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2, pos.y1 + optical_caption_h / fig_h, caption,
            fontsize=optical_size, ha="center", va="top",
        )

    _draw_activity_map(sar_ax, legend_ax, array, polygon, extent)
    activity_top = y_sar + sar_h + activity_caption_h
    area_top = activity_top - activity_size / 72.0 - text_gap
    sar_pos = sar_ax.get_position()
    fig.text(
        sar_pos.x0 + sar_pos.width / 2, activity_top / fig_h, "Activity Map  |  %s" % dates,
        fontsize=activity_size, ha="center", va="top",
    )
    fig.text(
        sar_pos.x0 + sar_pos.width / 2, area_top / fig_h, area_changed,
        fontsize=area_size, ha="center", va="top",
    )

    out = OUTPUT_DIR / ("%s-%s_activity_map_optical_images.png" % (master_day.strftime("%Y.%m.%d"), slave_day.strftime("%Y.%m.%d")))
    fig.savefig(out, facecolor="white")
    print("Saved map: %s" % out)
    plt.close(fig)
    return out


# --- notebook cell 44 ---
def _java_field(obj, name):
    unsafe = _unsafe()
    cls = obj.getClass()
    while cls is not None:
        try:
            field = cls.getDeclaredField(name)
        except Exception:
            cls = cls.getSuperclass()
            continue
        try:
            field.setAccessible(True)
            return field.get(obj)
        except Exception:
            return unsafe.getObject(obj, unsafe.objectFieldOffset(field))
    return None


def _unsafe():
    _ensure_snap()
    Class = jpy.get_type("java.lang.Class")
    unsafe_cls = Class.forName("sun.misc.Unsafe")
    unsafe_field = unsafe_cls.getDeclaredField("theUnsafe")
    unsafe_field.setAccessible(True)
    return unsafe_field.get(None)


def _release_product(product):
    reader = product.getProductReader()
    data_dir = _java_field(reader, "dataDir")
    if data_dir is not None:
        _java_field(_java_field(data_dir, "productDir"), "zipFile").close()
    product.dispose()


def _release_zip_sources(path):
    unsafe = _unsafe()
    Class = jpy.get_type("java.lang.Class")
    source_cls = Class.forName("java.util.zip.ZipFile$Source")
    files_field = source_cls.getDeclaredField("files")
    files = unsafe.getObject(unsafe.staticFieldBase(files_field), unsafe.staticFieldOffset(files_field))
    refs_offset = unsafe.objectFieldOffset(source_cls.getDeclaredField("refs"))
    zfile_offset = unsafe.objectFieldOffset(source_cls.getDeclaredField("zfile"))
    target = str(Path(path).resolve()).lower()
    
    for key in files.keySet().toArray():
        file_obj = unsafe.getObject(key, unsafe.objectFieldOffset(key.getClass().getDeclaredField("file")))
        if str(file_obj.getCanonicalPath()).lower() != target:
            continue
        src = files.remove(key)
        raf = unsafe.getObject(src, zfile_offset)
        raf.close()
        unsafe.putObject(src, zfile_offset, None)
        unsafe.putInt(src, refs_offset, 0)


def close_products(products):
    already_closed = set()
    for product in reversed(products):
        identity = int(product.hashCode())
        if identity in already_closed:
            continue
        already_closed.add(identity)
        _release_product(product)


def remove_scene(path):
    _release_zip_sources(path)
    path.unlink()
    print("Removed: %s" % path.name)


# --- notebook cell 46 functions ---

def coherence_analysis(master_zip, slave_zip):
    _ensure_snap()
    master_slc = open_slc(master_zip)
    slave_slc = open_slc(slave_zip)

    master_orb = apply_orbit_file(master_slc)
    slave_orb = apply_orbit_file(slave_slc)

    subswath = choose_subswaths_for_aoi(master_orb)

    master_orb_split = { swath: topsar_split(master_orb, swath) for swath in subswath }
    slave_orb_split = { swath: topsar_split(slave_orb, swath) for swath in subswath }

    orb_split_stack = { swath: back_geocoding(master_orb_split[swath], slave_orb_split[swath]) for swath in subswath }
    orb_split_stack_coh = { swath: coherence_estimation(orb_split_stack[swath]) for swath in subswath }
    orb_split_stack_coh_deb = { swath: topsar_deburst(orb_split_stack_coh[swath]) for swath in subswath }

    if len(orb_split_stack_coh_deb) > 1:
        orb_split_stack_coh_deb_merge = topsar_merge([orb_split_stack_coh_deb[swath] for swath in subswath])
    else:
        orb_split_stack_coh_deb_merge = orb_split_stack_coh_deb[subswath[0]]

    orb_split_stack_coh_deb_merge_multi = multilook(orb_split_stack_coh_deb_merge)
    orb_split_stack_coh_deb_merge_multi_tc = terrain_correction(orb_split_stack_coh_deb_merge_multi)

    mine_sub = subset(orb_split_stack_coh_deb_merge_multi_tc, AOI_MINE)
    stable_points_sub = subset(orb_split_stack_coh_deb_merge_multi_tc, AOI_STABLE_POINTS)

    mine = raster_to_array(mine_sub, master_zip, slave_zip, "mine")
    stable_points = raster_to_array(stable_points_sub, master_zip, slave_zip, "stable_points")

    opened = [
        master_slc, slave_slc, master_orb, slave_orb,
        *master_orb_split.values(), *slave_orb_split.values(),
        *orb_split_stack.values(),
        *orb_split_stack_coh.values(),
        *orb_split_stack_coh_deb.values(),
        orb_split_stack_coh_deb_merge,
        orb_split_stack_coh_deb_merge_multi, 
        orb_split_stack_coh_deb_merge_multi_tc, 
        mine_sub, stable_points_sub,
    ]
    
    close_products(opened)

    return {"mine": mine, "stable_points": stable_points}


def stable_points_analysis(references):
    building_mask = buildings_in_reference_window(references)
    covered = pixels_in_enough_pairs(references, building_mask)
    strict, mean, std = stable_pixels(references, covered)
    stable = compact_stable_pixels(strict)
    return gamma_stable_per_pair(references, stable, mean, std)


def ndai_samples(image, reference, by_pair, extent, shape_hw, grid_transform, mine_mask, master_day, slave_day, samples, prepared):
    raster = by_pair[(reference["master_day"], reference["slave_day"])]
    cloud = cloud_mask(image, extent, shape_hw, grid_transform)
    vegetation = ndvi_swir_mask(image, extent, shape_hw, grid_transform, cloud)
    water = water_mask(image, extent, shape_hw, grid_transform, vegetation, cloud)
    shadow = shadow_mask(raster, shape_hw)
    ndai_array = ndai_index(raster["coherence"], reference["gamma_stable"])

    sample = ndai_samples_inside_mine(ndai_array, mine_mask, water, vegetation, shadow, cloud)
    if sample.size:
        samples.append(sample)
    prepared.append({
        "master_day": master_day,
        "slave_day": slave_day,
        "ndai": ndai_array,
        "shadow": shadow,
        "water": water,
        "vegetation": vegetation,
        "cloud": cloud,
    })

def stable_points_analysis(references):
    building_mask = buildings_in_reference_window(references)
    covered = pixels_in_enough_pairs(references, building_mask)
    strict, mean, std = stable_pixels(references, covered)
    stable = compact_stable_pixels(strict)
    return gamma_stable_per_pair(references, stable, mean, std)


def ndai_samples(image, reference, by_pair, extent, shape_hw, grid_transform, mine_mask, master_day, slave_day, samples, prepared):
    raster = by_pair[(reference["master_day"], reference["slave_day"])]
    cloud = cloud_mask(image, extent, shape_hw, grid_transform)
    vegetation = ndvi_swir_mask(image, extent, shape_hw, grid_transform, cloud)
    water = water_mask(image, extent, shape_hw, grid_transform, vegetation, cloud)
    shadow = shadow_mask(raster, shape_hw)
    ndai_array = ndai_index(raster["coherence"], reference["gamma_stable"])

    sample = ndai_samples_inside_mine(ndai_array, mine_mask, water, vegetation, shadow, cloud)
    if sample.size:
        samples.append(sample)
    prepared.append({
        "master_day": master_day,
        "slave_day": slave_day,
        "ndai": ndai_array,
        "shadow": shadow,
        "water": water,
        "vegetation": vegetation,
        "cloud": cloud,
    })

def ndai_samples(image, reference, by_pair, extent, shape_hw, grid_transform, mine_mask, master_day, slave_day, samples, prepared):
    raster = by_pair[(reference["master_day"], reference["slave_day"])]
    cloud = cloud_mask(image, extent, shape_hw, grid_transform)
    vegetation = ndvi_swir_mask(image, extent, shape_hw, grid_transform, cloud)
    water = water_mask(image, extent, shape_hw, grid_transform, vegetation, cloud)
    shadow = shadow_mask(raster, shape_hw)
    ndai_array = ndai_index(raster["coherence"], reference["gamma_stable"])

    sample = ndai_samples_inside_mine(ndai_array, mine_mask, water, vegetation, shadow, cloud)
    if sample.size:
        samples.append(sample)
    prepared.append({
        "master_day": master_day,
        "slave_day": slave_day,
        "ndai": ndai_array,
        "shadow": shadow,
        "water": water,
        "vegetation": vegetation,
        "cloud": cloud,
    })
