# open-pit-snap

CLI port of Zuzanna Słobodzian’s `mine_analysis.ipynb`.

- **Sentinel-1** (`period.start` before 2026-06-17, or `--sensor sentinel1`): full **SAFE SLC** + **SNAP / esa_snappy** (Orbit → TOPS Split → Back-Geocoding → Coherence **10×3** → Deburst → Merge → Multilook **6×1** → Terrain Correction → Subset).
- **NISAR** (`period.start` on/after **2026-06-17**, or `--sensor nisar`): already-geocoded **L-band GSLC** subsets (no SAFE/SNAP), boxcar coherence **10×3**, then the same GEE/S2/NDAI activity maps.

`init --sensor auto` (default) picks NISAR vs S1 from the period. Counterpart: `open-pit-baseline` (bursts + COMPASS / NISAR GSLC).

## Requirements

- Windows with **SNAP** + **esa_snappy** (Python linked to SNAP)
- Python 3.9+
- Earthdata token (`token_asf`) and Earth Engine project id (`gee_project`)

```bash
cd c:\space_technologies\Kopalnie\open-pit-snap
python -m pip install -e .
# NISAR path also needs:
python -m pip install -e ".[nisar]"
```

Place `token_asf` / `gee_project` in this folder (or reuse the files from `open-pit-baseline` / `open-pit-mining\sar`).

## Lubienica comparison project

Already configured at `C:\SAR_MINING\lubienica_snap` for **2024-03-01 … 2024-05-10** (same window as the open-pit-baseline timing run).

**Run on Windows Python with SNAP/esa_snappy** (not WSL — SNAP is a Windows install). In PowerShell:

```powershell
cd c:\space_technologies\Kopalnie\open-pit-snap
python -m pip install -e .
open-pit-snap pipeline --root C:\SAR_MINING\lubienica_snap
```

In **Git Bash** use a Windows-style path (forward slashes are fine):

```bash
open-pit-snap pipeline --root C:/SAR_MINING/lubienica_snap
```

In **WSL** use `/mnt/c/...`. Do not mix `/mnt/c` with Windows Python from Git Bash.

Step by step:

```powershell
open-pit-snap catalog --root C:\SAR_MINING\lubienica_snap
open-pit-snap coherence --root C:\SAR_MINING\lubienica_snap
```

(`coherence` downloads SAFE zips as needed and runs SNAP; activity maps are included in `pipeline`.)

## Outputs (flat under the site root)

```
lubienica_snap/
  project.yaml
  raw/                 # SAFE .zip  or  NISAR GSLC .h5 subsets
  coherence/
    mine/              # GeoTIFF coherence (+ layover/shadow when SNAP)
    reference/
  activity_maps/       # PNG maps (notebook style)
```

Nothing else is written (`catalog` / `coherence_pairs.json` are not saved).

## New site

```powershell
open-pit-snap init --root C:\SAR_MINING\my_mine `
  --mine C:\path\to\mine.geojson `
  --start 2024-03-01 --end 2024-05-10 `
  --reference-center LON LAT `
  --voivodeship mazowieckie
open-pit-snap pipeline --root C:\SAR_MINING\my_mine
```
