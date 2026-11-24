# open-pit-snap

## Run

```powershell
open-pit-snap init --root C:\SAR_MINING\my_mine `
  --mine C:\path\to\mine.geojson `
  --start 2024-03-01 --end 2024-05-10 `
  --reference-center LON LAT `
  --voivodeship mazowieckie
open-pit-snap pipeline --root C:\SAR_MINING\my_mine
```

`init --sensor auto` (default) picks NISAR vs S1 from the period.

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
`

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