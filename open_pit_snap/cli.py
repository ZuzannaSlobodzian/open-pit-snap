"""CLI for the notebook-faithful SNAP/SAFE open-pit pipeline."""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from . import __version__
from .config import DEFAULTS, Project, resolve_gee_project, save_project


class _Tee:
    """Copy text written to a terminal stream into the project log."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _attach_log(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run.log"
    handle = open(path, "a", encoding="utf-8", errors="replace")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write("\n--- %s %s ---\n" % (stamp, " ".join(sys.argv)))
    handle.flush()
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    print("Log: %s" % path, flush=True)
    return path


def absolute(path: str) -> Path:
    """Resolve project paths across PowerShell, Git Bash, and WSL."""
    text = path.strip().strip('"').strip("'")
    # WSL/Linux: C:\foo → /mnt/c/foo
    win = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if win and os.name != "nt":
        drive, rest = win.group(1).lower(), win.group(2).replace("\\", "/")
        text = f"/mnt/{drive}/{rest}"
    # Windows (incl. Git Bash): /mnt/c/foo or /c/foo → C:\foo
    elif os.name == "nt":
        mnt = re.match(r"^/mnt/([A-Za-z])/(.*)$", text.replace("\\", "/"))
        msys = re.match(r"^/([A-Za-z])/(.*)$", text.replace("\\", "/"))
        hit = mnt or msys
        if hit:
            drive, rest = hit.group(1).upper(), hit.group(2).replace("/", "\\")
            text = f"{drive}:\\{rest}"
    return Path(text).expanduser().resolve()


def init(args) -> Path:
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    path = root / "project.yaml"
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    if not args.voivodeship:
        raise ValueError("--voivodeship is required")
    from .config import resolve_sensor

    period = {"start": args.start, "end": args.end}
    requested = args.sensor or "auto"
    resolved = resolve_sensor({"sensor": requested, "period": period})
    settings = {
        "schema_version": 1,
        "name": args.name or root.name,
        "mine": str(args.mine),
        "period": period,
        "sensor": requested,
        "grid": DEFAULTS["grid"],
        "pairs": DEFAULTS["pairs"],
        "reference": {
            **DEFAULTS["reference"],
            "center": [float(args.reference_center[0]), float(args.reference_center[1])],
            "voivodeship": args.voivodeship,
        },
        "quality": DEFAULTS["quality"],
        "optical": {
            "source": "earth-engine",
            "ee_project": args.ee_project or resolve_gee_project(),
        },
        "method": "nisar-gslc" if resolved == "nisar" else "snap-safe-notebook",
    }
    # Widen max pair gap slightly for NISAR 12-day repeats when user left defaults.
    if resolved == "nisar" and settings["pairs"]["max_days"] < 14:
        settings["pairs"]["max_days"] = 14
    save_project(root, settings)
    print(f"sensor: {resolved}", flush=True)
    return path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="open-pit-snap",
        description="Notebook-faithful SNAP/SAFE open-pit activity pipeline",
    )
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)

    item = commands.add_parser("init", help="Write project.yaml")
    item.add_argument("--root", type=absolute, required=True)
    item.add_argument("--name")
    item.add_argument("--mine", type=absolute, required=True)
    item.add_argument("--start", required=True)
    item.add_argument("--end", required=True)
    item.add_argument("--reference-center", nargs=2, type=float, metavar=("LON", "LAT"), required=True)
    item.add_argument("--voivodeship", required=True)
    item.add_argument("--ee-project")
    item.add_argument(
        "--sensor",
        choices=("auto", "sentinel1", "nisar"),
        default="auto",
        help="auto picks NISAR when period.start >= 2026-06-17 (skip SAFE/SNAP)",
    )

    for name, help_text in (
        ("catalog", "Search full SAFE SLC scenes covering the processing AOI"),
        ("download", "Download SAFE zips for the catalogued stack"),
        ("coherence", "Download if needed and run SNAP coherence for consecutive pairs"),
        ("activity", "Build NDAI activity maps from saved coherence GeoTIFFs"),
        ("pipeline", "catalog → SAFE download + SNAP coherence → activity maps"),
    ):
        item = commands.add_parser(name, help=help_text)
        item.add_argument("--root", type=absolute, required=True)
        item.add_argument("--earthdata-token", type=absolute)
        item.add_argument("--ee-project")

    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _attach_log(args.root)
    try:
        if args.command == "init":
            path = init(args)
            print(f"Project: {path}", flush=True)
            return 0

        # Imported after the log is attached, so Earth Engine warnings land in run.log.
        from .runtime import bind
        from .workflow import catalog, coherence, download, pipeline, run_activity

        project = Project.load(args.root)
        bind(project, earthdata_token=getattr(args, "earthdata_token", None),
             ee_project=getattr(args, "ee_project", None))

        if args.command == "catalog":
            catalog(project)
            return 0
        if args.command == "download":
            download(project)
            return 0
        if args.command == "coherence":
            scenes, pin = catalog(project)
            mine, ref = coherence(project, scenes, pin)
            return 0 if mine else 1
        if args.command == "activity":
            mapped = run_activity(project)
            return 0 if mapped else 1
        if args.command == "pipeline":
            return pipeline(
                project,
                earthdata_token=getattr(args, "earthdata_token", None),
                ee_project=getattr(args, "ee_project", None),
            )
        raise ValueError(f"Unknown command {args.command}")
    except Exception as exc:
        print(f"open-pit-snap: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
