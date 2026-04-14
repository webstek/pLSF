#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# benchmark.py – Filtration time comparison: pLSF vs Cubical Ripser vs GUDHI
#                vs DIPHA
# ──────────────────────────────────────────────────────────────────────────────
"""
Compare filtration construction time of pLSF (GPU), Cubical Ripser (CPU),
GUDHI (CPU), and DIPHA (CPU/MPI) on 3-D NIfTI volumes.

Accepts one or more .nii files and produces:
  • A summary table per file
  • An optional plot of filtration time vs number of cells across files

Each tool can be skipped with ``--skip-<tool>``.  If a tool errors on a
particular file the other tools still run.

Usage:
    python benchmark.py data/*.nii [options]
    python benchmark.py data/small.nii data/large.nii --plot filtration.png

Dependencies:
    pip install nibabel numpy gudhi matplotlib
    External: plsf (this project),
              cubicalripser (lib/CubicalRipser, built with --filtration-only),
              dipha (lib/dipha, built with --filtration-only)
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR.parent / "bin"
CR_BUILD_DIR = SCRIPT_DIR.parent / "lib" / "CubicalRipser" / "build"
DIPHA_BUILD_DIR = SCRIPT_DIR.parent / "lib" / "dipha" / "build"

# ── Helpers ──────────────────────────────────────────────────────────────────


def which(name: str) -> str | None:
    return shutil.which(name)


def fmt_time(ms: float) -> str:
    if ms < 1000.0:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.3f} s"


def print_table(
    rows: list[tuple[str, str, str]], col_widths: tuple[int, ...] = (36, 18, 14)
) -> None:
    header = ("Step", "Pipeline", "Time")
    print()
    print("─" * sum(col_widths))
    print(
        f"{header[0]:<{col_widths[0]}}"
        f"{header[1]:<{col_widths[1]}}"
        f"{header[2]:>{col_widths[2]}}"
    )
    print("─" * sum(col_widths))
    for label, pipeline, elapsed in rows:
        print(
            f"{label:<{col_widths[0]}}"
            f"{pipeline:<{col_widths[1]}}"
            f"{elapsed:>{col_widths[2]}}"
        )
    print("─" * sum(col_widths))
    print()


def load_nii_array(nii_path: Path):
    """Load a NIfTI file and return (data_array, load_time_ms)."""
    import nibabel as nib
    import numpy as np

    t0 = time.perf_counter()
    img = nib.load(str(nii_path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    load_ms = (time.perf_counter() - t0) * 1000.0
    return data, load_ms


def count_cells_from_shape(shape: tuple[int, ...]) -> int:
    """Estimate total number of cells in a cubical complex from the grid shape.

    For a d-dimensional grid of shape (n1, n2, ..., nd), the number of cells
    is the product of (2*ni - 1) for each dimension (the full cubical complex
    grid).
    """
    result = 1
    for n in shape:
        result *= 2 * n - 1
    return result


def _parse_mem_mib(s: str) -> float:
    """Parse a memory string like '366 MiB' or '2 GiB' into MiB.  Return 0 for '-'."""
    s = s.strip()
    if s == "-":
        return 0.0
    m = re.match(r"(\d+)\s*(KiB|MiB|GiB)", s)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "KiB":
        return val / 1024.0
    if unit == "GiB":
        return val * 1024.0
    return val  # MiB


def _get_current_rss_mib() -> float:
    """Return current process RSS in MiB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _get_peak_rss_from_cmd(cmd: list[str]) -> float:
    """Run a command under /usr/bin/time and return its peak RSS in MiB."""
    try:
        r = subprocess.run(
            ["/usr/bin/time", "-v"] + cmd,
            capture_output=True, text=True,
        )
        for line in r.stderr.splitlines():
            if "Maximum resident set size" in line:
                # Value is in KiB
                return float(line.split()[-1]) / 1024.0
    except Exception:
        pass
    return 0.0


# ── pLSF filtration-only timing ─────────────────────────────────────────────


def run_plsf_filtration(
    nii_path: Path, plsf_bin: str, device: str, verbose: bool,
    compress: bool = False, lossy: bool = False
) -> dict:
    """Run ``plsf -f -t`` and parse its table output."""
    cmd = [plsf_bin, str(nii_path), "-f", "-t", "-d", device]
    if compress:
        cmd.append("-x")
    if lossy:
        cmd.append("-l")
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"plsf failed (exit {result.returncode}):\n{result.stderr}")

    timings: dict[str, float] = {}
    host_rss: dict[str, float] = {}
    dev_peak: dict[str, float] = {}
    for line in result.stdout.splitlines():
        m = re.match(
            r"^(\S[\w\s]+?)\s{2,}([\d.]+)\s*ms"
            r"\s+(\d+\s*(?:KiB|MiB|GiB)|-)"
            r"\s+(\d+\s*(?:KiB|MiB|GiB)|-)",
            line,
        )
        if m:
            label = m.group(1).strip()
            timings[label] = float(m.group(2))
            host_rss[label] = _parse_mem_mib(m.group(3))
            dev_peak[label] = _parse_mem_mib(m.group(4))
        else:
            m2 = re.match(r"^(\S[\w\s]+?)\s{2,}([\d.]+)\s*ms", line)
            if m2:
                label = m2.group(1).strip()
                timings[label] = float(m2.group(2))

    # Take max across all steps for peak values
    peak_host_mib = max(host_rss.values(), default=0.0)
    peak_dev_mib = max(dev_peak.values(), default=0.0)

    return {
        "load_ms": timings.get("NIfTI load", 0.0),
        "enum_ms": timings.get("Complex computation", 0.0),
        "sort_ms": timings.get("Complex sorting", 0.0),
        "peak_host_mib": peak_host_mib,
        "peak_dev_mib": peak_dev_mib,
        "stdout": result.stdout,
    }


# ── Cubical Ripser filtration-only timing ────────────────────────────────────


def nii_to_npy_f64(nii_path: Path, npy_path: Path) -> float:
    """Convert a NIfTI file to a float64 .npy and return the time taken."""
    import nibabel as nib
    import numpy as np

    t0 = time.perf_counter()
    img = nib.load(str(nii_path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    np.save(str(npy_path), data)
    return (time.perf_counter() - t0) * 1000.0  # ms


def run_cr_filtration(
    npy_path: Path, cr_bin: str, maxdim: int, verbose: bool
) -> dict:
    """Run cubicalripser ``--filtration-only`` and parse TIMING lines."""
    cmd = [cr_bin, "--filtration-only", "-m", str(maxdim), str(npy_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cubicalripser failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Measure peak RSS via /proc on Linux
    peak_host_mib = _get_peak_rss_from_cmd(cmd)

    load_ms = 0.0
    total_enum_ms = 0.0
    total_sort_ms = 0.0
    dim_timings: list[dict] = []

    for line in result.stdout.splitlines():
        if not line.startswith("TIMING:"):
            continue
        tokens = dict(
            tok.split("=", 1) for tok in line.split() if "=" in tok
        )
        if "load_ms" in tokens and "dim" not in tokens:
            load_ms = float(tokens["load_ms"])
        elif "dim" in tokens:
            e = float(tokens.get("enum_ms", 0))
            s = float(tokens.get("sort_ms", 0))
            total_enum_ms += e
            total_sort_ms += s
            dim_timings.append(
                {
                    "dim": int(tokens["dim"]),
                    "enum_ms": e,
                    "sort_ms": s,
                    "cells": int(tokens.get("cells", 0)),
                }
            )

    return {
        "load_ms": load_ms,
        "enum_ms": total_enum_ms,
        "sort_ms": total_sort_ms,
        "dim_timings": dim_timings,
        "peak_host_mib": peak_host_mib,
        "stdout": result.stdout,
    }


# ── GUDHI filtration timing ─────────────────────────────────────────────────


def run_gudhi_filtration(data, verbose: bool) -> dict:
    """Time GUDHI CubicalComplex creation and persistence on an array.

    ``data`` should be a NumPy float64 array (already loaded from NIfTI).
    We time only the CubicalComplex construction and `.persistence()` call,
    which internally builds and sorts the filtration.
    """
    import gudhi as gd

    rss_before = _get_current_rss_mib()

    t0 = time.perf_counter()
    cc = gd.CubicalComplex(top_dimensional_cells=data)
    construct_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    cc.persistence(homology_coeff_field=2, min_persistence=0)
    persistence_ms = (time.perf_counter() - t1) * 1000.0

    rss_after = _get_current_rss_mib()
    peak_host_mib = max(rss_after - rss_before, 0.0)

    total_ms = construct_ms + persistence_ms

    if verbose:
        print(f"  GUDHI construct: {fmt_time(construct_ms)}")
        print(f"  GUDHI persistence: {fmt_time(persistence_ms)}")
        print(f"  GUDHI total: {fmt_time(total_ms)}")

    return {
        "construct_ms": construct_ms,
        "persistence_ms": persistence_ms,
        "total_ms": total_ms,
        "peak_host_mib": peak_host_mib,
    }


# ── DIPHA filtration-only timing ────────────────────────────────────────────


def nii_to_dipha_complex(nii_path: Path, complex_path: Path) -> float:
    """Convert a NIfTI file to a DIPHA .complex and return time in ms."""
    import struct
    import nibabel as nib
    import numpy as np

    _DIPHA_MAGIC = 8067171840
    _DIPHA_COMPLEX_TYPE = 1

    t0 = time.perf_counter()
    img = nib.load(str(nii_path))
    data = np.asarray(img.dataobj, dtype=np.float64)

    # DIPHA expects column-major (Fortran) ordering for 3-D data
    shape = data.shape
    ndim = len(shape)
    size = int(np.prod(shape))
    payload = data
    if ndim == 3:
        payload = data.transpose((2, 1, 0))
    elif ndim == 2:
        payload = data.transpose((1, 0))

    with complex_path.open("wb") as f:
        f.write(struct.pack("qqqq", _DIPHA_MAGIC, _DIPHA_COMPLEX_TYPE, size, ndim))
        f.write(struct.pack("q" * ndim, *shape))
        f.write(payload.ravel().tobytes())

    return (time.perf_counter() - t0) * 1000.0


def run_dipha_filtration(
    complex_path: Path, dipha_bin: str, verbose: bool,
    nprocs: int = 1,
) -> dict:
    """Run ``mpirun -n <nprocs> dipha --filtration-only`` and parse TIMING lines."""
    # Use a dummy output file (DIPHA requires an output argument)
    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as f:
        dummy_out = f.name

    cmd = ["mpirun", "--allow-run-as-root", "-n", str(nprocs),
           dipha_bin, "--filtration-only", str(complex_path), dummy_out]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up dummy output
    Path(dummy_out).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"dipha failed (exit {result.returncode}):\n{result.stderr}"
        )

    load_ms = 0.0
    filtration_ms = 0.0
    cells = 0
    peak_mem_mib = 0.0

    for line in result.stdout.splitlines():
        if not line.startswith("TIMING:"):
            continue
        tokens = dict(
            tok.split("=", 1) for tok in line.split() if "=" in tok
        )
        if "load_ms" in tokens:
            load_ms = float(tokens["load_ms"])
        if "filtration_ms" in tokens:
            filtration_ms = float(tokens["filtration_ms"])
        if "cells" in tokens:
            cells = int(tokens["cells"])
        if "peak_mem_mib" in tokens:
            peak_mem_mib = float(tokens["peak_mem_mib"])

    if verbose:
        print(result.stdout)

    return {
        "load_ms": load_ms,
        "filtration_ms": filtration_ms,
        "cells": cells,
        "peak_host_mib": peak_mem_mib,
        "stdout": result.stdout,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def _probe_hardware() -> dict:
    """Return a dict with CPU and GPU info strings for the hardware summary."""
    import subprocess

    cpu_name = "Unknown CPU"
    cpu_cores = "?"
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in lscpu.splitlines():
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "Model name":
                cpu_name = v
            elif k == "CPU(s)" and cpu_cores == "?":
                cpu_cores = v
    except Exception:
        pass

    gpu_name = "Unknown GPU"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
        if out and out[0]:
            gpu_name = out[0]
    except Exception:
        pass

    return {"cpu_name": cpu_name, "cpu_cores": cpu_cores, "gpu_name": gpu_name}


def make_plot(
    results: list[dict],
    output_path: Path,
    dipha_nodes: int = 1,
) -> None:
    """Plot filtration time and peak memory vs number of voxels (side by side).

    ``results`` is a list of dicts with keys:
        "file", "num_voxels", "num_cells",
        "plsf_total_ms", "cr_total_ms", "gudhi_total_ms", "dipha_total_ms",
        "plsf_host_mib", "plsf_dev_mib", "cr_host_mib", "gudhi_host_mib",
        "dipha_host_mib"
    (any may be None if skipped or failed).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: r["num_voxels"])
    voxels = [r["num_voxels"] for r in results]

    hw = _probe_hardware()

    # ── Tool style definitions ───────────────────────────────────────────
    # (time_key, mem_key, legend_label, linestyle, marker)
    tool_styles = [
        ("plsf_total_ms",  "plsf_host_mib",  "pLSF (host)",    "-",  "o"),
        (None,              "plsf_dev_mib",   "pLSF (device)",  "-",  "x"),
        ("cr_total_ms",    "cr_host_mib",     "Cubical Ripser", "--", "s"),
        ("gudhi_total_ms", "gudhi_host_mib",  "GUDHI",          "-.", "^"),
        ("dipha_total_ms", "dipha_host_mib",  "DIPHA",          ":",  "D"),
    ]

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: timing ───────────────────────────────────────────────
    for time_key, _mem_key, label, ls, marker in tool_styles:
        if time_key is None:
            continue
        xs, ys = [], []
        for r, v in zip(results, voxels):
            val = r.get(time_key)
            if val is not None:
                xs.append(v)
                ys.append(val / 1000.0)
        if xs:
            ax_time.plot(xs, ys, linestyle=ls, marker=marker, color="black",
                         label=label, linewidth=1.5, markersize=7)

    ax_time.set_xlabel("Number of voxels")
    ax_time.set_ylabel("Total filtration time (s)  [includes file load]")
    ax_time.set_title("Filtration Time")
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)

    if len(voxels) > 1 and max(voxels) / max(min(voxels), 1) > 10:
        ax_time.set_xscale("log")
    ys_all = [
        r.get(k) / 1000.0
        for r in results
        for k, *_ in tool_styles
        if k is not None and r.get(k) is not None
    ]
    if ys_all and max(ys_all) / max(min(ys_all), 1e-9) > 10:
        ax_time.set_yscale("log")

    # ── Right panel: peak memory ─────────────────────────────────────────
    for _time_key, mem_key, label, ls, marker in tool_styles:
        xs, ys = [], []
        for r, v in zip(results, voxels):
            val = r.get(mem_key)
            if val is not None and val > 0:
                xs.append(v)
                ys.append(val)
        if xs:
            ax_mem.plot(xs, ys, linestyle=ls, marker=marker, color="black",
                        label=label, linewidth=1.5, markersize=7)

    ax_mem.set_xlabel("Number of voxels")
    ax_mem.set_ylabel("Peak memory (MiB)")
    ax_mem.set_title("Peak Memory Usage")
    ax_mem.legend(fontsize=8)
    ax_mem.grid(True, alpha=0.3)

    if len(voxels) > 1 and max(voxels) / max(min(voxels), 1) > 10:
        ax_mem.set_xscale("log")
    mem_all = [
        r.get(mk)
        for r in results
        for _, mk, *_ in tool_styles
        if r.get(mk) is not None and r.get(mk) > 0
    ]
    if mem_all and max(mem_all) / max(min(mem_all), 1e-9) > 10:
        ax_mem.set_yscale("log")

    # ── Hardware summary ─────────────────────────────────────────────────
    dipha_hw = f"DIPHA          \u2014 {hw['cpu_name']}, {hw['cpu_cores']} cores"
    if dipha_nodes > 1:
        dipha_hw += f", {dipha_nodes} MPI processes"
    hw_lines = [
        f"pLSF           \u2014 {hw['gpu_name']} (CUDA)",
        f"Cubical Ripser \u2014 {hw['cpu_name']}, {hw['cpu_cores']} cores",
        f"GUDHI          \u2014 {hw['cpu_name']}, {hw['cpu_cores']} cores",
        dipha_hw,
    ]
    hw_text = "Hardware:  " + hw_lines[0]
    for line in hw_lines[1:]:
        hw_text += "\n           " + line
    fig.text(
        0.5, -0.02, hw_text,
        ha="center", va="top",
        fontsize=7.5,
        fontfamily="monospace",
        transform=fig.transFigure,
    )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


# ── Per-file benchmark ───────────────────────────────────────────────────────


def benchmark_one_file(nii_path: Path, args) -> dict:
    """Run all enabled tools on a single NIfTI file.

    Returns a result dict with timing entries (None = skipped/failed).
    """
    import numpy as np

    print(f"\n{'═' * 68}")
    print(f"  File: {nii_path}")
    print(f"{'═' * 68}")

    # Load once to get shape / num_cells (shared across tools)
    data, load_ms_shared = load_nii_array(nii_path)
    shape = data.shape
    num_voxels = int(data.size)
    num_cells = count_cells_from_shape(shape)
    print(f"  Shape: {shape}   Voxels: {num_voxels:,}   Elementary cubes: {num_cells:,}")

    rows: list[tuple[str, str, str]] = []
    result: dict = {
        "file": str(nii_path),
        "shape": shape,
        "num_voxels": num_voxels,
        "num_cells": num_cells,
        "plsf_total_ms": None,
        "cr_total_ms": None,
        "gudhi_total_ms": None,
        "dipha_total_ms": None,
        # Memory (MiB) — None means skipped/failed
        "plsf_host_mib": None,
        "plsf_dev_mib": None,
        "cr_host_mib": None,
        "gudhi_host_mib": None,
        "dipha_host_mib": None,
    }

    # ── pLSF ─────────────────────────────────────────────────────────────
    if not args.skip_plsf:
        plsf_bin = args.plsf
        if not Path(plsf_bin).exists() and not which(plsf_bin):
            print(f"  [SKIP] plsf not found at {plsf_bin}", file=sys.stderr)
        else:
            try:
                info = run_plsf_filtration(
                    nii_path, plsf_bin, args.device, args.verbose,
                    compress=args.compress, lossy=args.lossy,
                )
                if args.verbose:
                    print(info["stdout"])

                rows.append(("File load", "pLSF", fmt_time(info["load_ms"])))
                rows.append(("Filtration construction", "pLSF", fmt_time(info["enum_ms"])))
                rows.append(("Filtration sorting", "pLSF", fmt_time(info["sort_ms"])))
                plsf_filt_ms = info["load_ms"] + info["enum_ms"] + info["sort_ms"]
                rows.append(("Total (load + filt)", "pLSF", fmt_time(plsf_filt_ms)))
                result["plsf_total_ms"] = plsf_filt_ms
                result["plsf_host_mib"] = info["peak_host_mib"]
                result["plsf_dev_mib"] = info["peak_dev_mib"]
            except Exception as e:
                print(f"  [ERROR] pLSF: {e}", file=sys.stderr)

    # ── Cubical Ripser ───────────────────────────────────────────────────
    if not args.skip_cripser:
        cr_bin = args.cubicalripser
        if not Path(cr_bin).exists() and not which(cr_bin):
            print(f"  [SKIP] cubicalripser not found at {cr_bin}", file=sys.stderr)
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="cr_bench_") as tmpdir:
                    npy_path = Path(tmpdir) / "input.npy"
                    conv_ms = nii_to_npy_f64(nii_path, npy_path)

                    rows.append(("NIfTI -> npy conversion", "CubicalRipser", fmt_time(conv_ms)))

                    cr = run_cr_filtration(npy_path, cr_bin, args.maxdim, args.verbose)
                    if args.verbose:
                        print(cr["stdout"])

                    rows.append(("File load (.npy)", "CubicalRipser", fmt_time(cr["load_ms"])))

                    for dt in cr["dim_timings"]:
                        d = dt["dim"]
                        rows.append((
                            f"Dim-{d} enumeration ({dt['cells']} cells)",
                            "CubicalRipser",
                            fmt_time(dt["enum_ms"]),
                        ))
                        rows.append((
                            f"Dim-{d} sorting",
                            "CubicalRipser",
                            fmt_time(dt["sort_ms"]),
                        ))

                    cr_filt_ms = conv_ms + cr["load_ms"] + cr["enum_ms"] + cr["sort_ms"]
                    rows.append(("Total (conv + load + filt)", "CubicalRipser", fmt_time(cr_filt_ms)))
                    result["cr_total_ms"] = cr_filt_ms
                    result["cr_host_mib"] = cr["peak_host_mib"]
            except Exception as e:
                print(f"  [ERROR] CubicalRipser: {e}", file=sys.stderr)

    # ── GUDHI ────────────────────────────────────────────────────────────
    if not args.skip_gudhi:
        try:
            import gudhi  # noqa: F401
        except ImportError:
            print("  [SKIP] GUDHI not installed (pip install gudhi)", file=sys.stderr)
        else:
            try:
                # data was already loaded above; charge the shared load time
                gudhi_info = run_gudhi_filtration(data, args.verbose)
                rows.append(("File load (shared)", "GUDHI", fmt_time(load_ms_shared)))
                rows.append(("CubicalComplex construction", "GUDHI", fmt_time(gudhi_info["construct_ms"])))
                rows.append(("Persistence computation", "GUDHI", fmt_time(gudhi_info["persistence_ms"])))
                gudhi_total = load_ms_shared + gudhi_info["total_ms"]
                rows.append(("Total (load + filt)", "GUDHI", fmt_time(gudhi_total)))
                result["gudhi_total_ms"] = gudhi_total
                result["gudhi_host_mib"] = gudhi_info["peak_host_mib"]
            except Exception as e:
                print(f"  [ERROR] GUDHI: {e}", file=sys.stderr)

    # ── DIPHA ────────────────────────────────────────────────────────────
    if not args.skip_dipha:
        dipha_bin = args.dipha
        if not Path(dipha_bin).exists() and not which(dipha_bin):
            print(f"  [SKIP] dipha not found at {dipha_bin}", file=sys.stderr)
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="dipha_bench_") as tmpdir:
                    complex_path = Path(tmpdir) / "input.complex"
                    conv_ms = nii_to_dipha_complex(nii_path, complex_path)

                    rows.append(("NIfTI -> .complex conversion", "DIPHA", fmt_time(conv_ms)))

                    dipha_info = run_dipha_filtration(
                        complex_path, dipha_bin, args.verbose,
                        nprocs=args.dipha_nodes,
                    )

                    rows.append(("File load (.complex)", "DIPHA", fmt_time(dipha_info["load_ms"])))
                    rows.append(("Filtration ordering", "DIPHA", fmt_time(dipha_info["filtration_ms"])))

                    dipha_total = conv_ms + dipha_info["load_ms"] + dipha_info["filtration_ms"]
                    rows.append(("Total (conv + load + filt)", "DIPHA", fmt_time(dipha_total)))
                    result["dipha_total_ms"] = dipha_total
                    result["dipha_host_mib"] = dipha_info["peak_host_mib"]
            except Exception as e:
                print(f"  [ERROR] DIPHA: {e}", file=sys.stderr)

    # ── Summary table ────────────────────────────────────────────────────
    print_table(rows)

    # Print speedup comparisons
    totals = {
        "pLSF": result["plsf_total_ms"],
        "CubicalRipser": result["cr_total_ms"],
        "GUDHI": result["gudhi_total_ms"],
        "DIPHA": result["dipha_total_ms"],
    }
    active = {k: v for k, v in totals.items() if v is not None and v > 0}
    if len(active) >= 2:
        fastest_name = min(active, key=active.get)
        fastest_ms = active[fastest_name]
        for name, ms in sorted(active.items(), key=lambda x: x[1]):
            if name != fastest_name:
                ratio = ms / fastest_ms
                print(
                    f"  {fastest_name} is {ratio:.2f}x faster than {name} "
                    f"({fmt_time(fastest_ms)} vs {fmt_time(ms)})"
                )

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare filtration construction time of pLSF (GPU), "
            "Cubical Ripser (CPU), GUDHI (CPU), and DIPHA (CPU/MPI) on "
            "3-D NIfTI volumes.\n\n"
            "Accepts one or more .nii files.  When multiple files are given, "
            "all enabled tools are benchmarked on each file and an optional "
            "plot of filtration time vs number of cells is produced."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # compare all tools on a single volume\n"
            "  python benchmark.py data/input.nii\n\n"
            "  # multiple files with a plot\n"
            "  python benchmark.py data/*.nii --plot filtration.png\n\n"
            "  # skip GUDHI and DIPHA\n"
            "  python benchmark.py data/input.nii --skip-gudhi --skip-dipha\n\n"
            "  # only pLSF\n"
            "  python benchmark.py data/input.nii --skip-cripser --skip-gudhi --skip-dipha\n"
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="INPUT",
        help=(
            "One or more NIfTI (.nii) files to benchmark. "
            "Each file is processed independently."
        ),
    )
    parser.add_argument(
        "--plsf",
        default=str(BIN_DIR / "plsf"),
        metavar="PATH",
        help="Path to the compiled plsf binary. Default: %(default)s",
    )
    parser.add_argument(
        "--cubicalripser",
        default=str(CR_BUILD_DIR / "cubicalripser"),
        metavar="PATH",
        help=(
            "Path to the compiled cubicalripser binary (V-construction). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--dipha",
        default=str(DIPHA_BUILD_DIR / "dipha"),
        metavar="PATH",
        help="Path to the compiled DIPHA binary. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "default"],
        help="CUDA device selector passed to plsf via -d. Default: %(default)s",
    )
    parser.add_argument(
        "--maxdim",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Maximum cell dimension for Cubical Ripser filtration. "
            "Default: %(default)s"
        ),
    )
    plsf_flags = parser.add_mutually_exclusive_group()
    plsf_flags.add_argument(
        "-x", "--compress",
        action="store_true",
        help="Pass -x to pLSF: use uint8_t filtration values.",
    )
    plsf_flags.add_argument(
        "-l", "--lossy",
        action="store_true",
        help="Pass -l to pLSF: lossy dimension encoding.",
    )
    parser.add_argument(
        "--skip-plsf",
        action="store_true",
        help="Do not run the pLSF pipeline.",
    )
    parser.add_argument(
        "--skip-cripser",
        action="store_true",
        help="Do not run the Cubical Ripser pipeline.",
    )
    parser.add_argument(
        "--skip-gudhi",
        action="store_true",
        help="Do not run the GUDHI pipeline.",
    )
    parser.add_argument(
        "--skip-dipha",
        action="store_true",
        help="Do not run the DIPHA pipeline.",
    )
    parser.add_argument(
        "--dipha-nodes",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of MPI processes (nodes) to launch DIPHA with via "
            "'mpirun -n N'. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save a plot of filtration time vs number of cells to this path "
            "(e.g. filtration.png). Requires matplotlib."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print raw subprocess output in addition to the summary table.",
    )
    args = parser.parse_args()

    # Validate inputs
    for inp in args.inputs:
        if not inp.exists():
            print(f"Error: {inp} not found", file=sys.stderr)
            sys.exit(1)

    # Run benchmarks
    all_results: list[dict] = []
    for nii_path in args.inputs:
        r = benchmark_one_file(nii_path, args)
        all_results.append(r)

    # Multi-file summary
    if len(all_results) > 1:
        print(f"\n{'═' * 68}")
        print("  Multi-file summary")
        print(f"{'═' * 68}")
        summary_header = f"{'File':<30} {'Voxels':>12} {'Elem. cubes':>14} {'pLSF':>10} {'CRipser':>10} {'GUDHI':>10} {'DIPHA':>10}"
        print(summary_header)
        print("─" * len(summary_header))
        for r in sorted(all_results, key=lambda x: x["num_cells"]):
            def _fmt(v):
                return fmt_time(v) if v is not None else "-"
            print(
                f"{Path(r['file']).name:<30} "
                f"{r['num_voxels']:>12,} "
                f"{r['num_cells']:>14,} "
                f"{_fmt(r['plsf_total_ms']):>10} "
                f"{_fmt(r['cr_total_ms']):>10} "
                f"{_fmt(r['gudhi_total_ms']):>10} "
                f"{_fmt(r['dipha_total_ms']):>10}"
            )
        print()

    # Generate plot
    if args.plot:
        try:
            make_plot(all_results, args.plot, dipha_nodes=args.dipha_nodes)
        except Exception as e:
            print(f"Error generating plot: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
