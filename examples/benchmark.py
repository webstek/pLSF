#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# benchmark.py – Filtration & sorting time comparison: pLSF vs Cubical Ripser
# ──────────────────────────────────────────────────────────────────────────────
"""
Compare the filtration construction and sorting phases of pLSF (GPU) against
Cubical Ripser (CPU) on 3-D NIfTI volumes.

pLSF is run with ``-f -t`` (filtration-only + timings) and its structured
table output is parsed.

Cubical Ripser is run with ``--filtration-only`` on a float64 .npy copy of
the same volume and its ``TIMING:`` lines are parsed.

Usage:
    python benchmark.py <input.nii> [options]

Dependencies:
    pip install nibabel numpy
    External: plsf (this project),
              cubicalripser (lib/CubicalRipser, built with --filtration-only)
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
        print(f"plsf failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse the table:
    #   Step                        Time     Host RSS     Dev peak
    #   ---------------------------------------------------------------
    #   NIfTI load                23.6 ms      136 MiB            -
    #   Complex computation      162.4 ms      433 MiB      295 MiB
    #   Complex sorting          216.2 ms      696 MiB      788 MiB
    timings: dict[str, float] = {}
    for line in result.stdout.splitlines():
        m = re.match(r"^(\S[\w\s]+?)\s{2,}([\d.]+)\s*ms", line)
        if m:
            label = m.group(1).strip()
            timings[label] = float(m.group(2))

    return {
        "load_ms": timings.get("NIfTI load", 0.0),
        "enum_ms": timings.get("Complex computation", 0.0),
        "sort_ms": timings.get("Complex sorting", 0.0),
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
        print(f"cubicalripser failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse lines like:
    #   TIMING: load_ms=232.474
    #   TIMING: dim=1 enum_ms=1244.06 sort_ms=1228.72 cells=25898696
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
        "stdout": result.stdout,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the filtration construction and sorting phases of pLSF "
            "(GPU) against Cubical Ripser (CPU) on a 3-D NIfTI volume.\n\n"
            "pLSF is invoked with -f -t (filtration-only + timings). Its output "
            "table is parsed for the 'Complex computation' and 'Complex sorting' "
            "rows.\n\n"
            "Cubical Ripser is invoked with --filtration-only on a float64 .npy "
            "copy of the same volume. Per-dimension enumeration and sorting times "
            "are parsed from TIMING: lines on stdout.\n\n"
            "Only filtration and sorting times are compared; PH reduction is not "
            "performed by either pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # compare both pipelines on a NIfTI volume\n"
            "  python benchmark.py data/input.nii\n\n"
            "  # use a specific device and show per-step output\n"
            "  python benchmark.py data/input.nii --device default -v\n\n"
            "  # run only Cubical Ripser (skip pLSF)\n"
            "  python benchmark.py data/input.nii --skip-plsf\n\n"
            "  # limit Cubical Ripser to H0 and H1\n"
            "  python benchmark.py data/input.nii --maxdim 1\n"
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to the input NIfTI file (.nii, uncompressed). "
            "Both pipelines operate on this volume."
        ),
    )
    parser.add_argument(
        "--plsf",
        default=str(BIN_DIR / "plsf"),
        metavar="PATH",
        help=(
            "Path to the compiled plsf binary. "
            "plsf computes the lower-star filtration on the GPU. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--cubicalripser",
        default=str(CR_BUILD_DIR / "cubicalripser"),
        metavar="PATH",
        help=(
            "Path to the compiled cubicalripser binary (V-construction). "
            "Must be built from lib/CubicalRipser with the --filtration-only "
            "patch applied (cmake --build lib/CubicalRipser/build). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "default"],
        help=(
            "CUDA device selector passed to plsf via -d. "
            "'gpu' selects the first discrete GPU; "
            "'default' lets the CUDA runtime choose. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--maxdim",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Maximum cell dimension whose filtration Cubical Ripser will "
            "enumerate and sort. For a 3-D volume the effective ceiling is 2 "
            "(dim-1), so values above 2 have no extra effect on 3-D inputs. "
            "Default: %(default)s"
        ),
    )
    plsf_flags = parser.add_mutually_exclusive_group()
    plsf_flags.add_argument(
        "-x", "--compress",
        action="store_true",
        help=(
            "Pass -x to pLSF: use uint8_t filtration values instead of the "
            "NIfTI native type to reduce GPU memory pressure. "
            "Incompatible with --lossy."
        ),
    )
    plsf_flags.add_argument(
        "-l", "--lossy",
        action="store_true",
        help=(
            "Pass -l to pLSF: encode cell dimension in the 2 LSBs of the "
            "sortable float key, enabling a single-pass sort "
            "(float/double only). Incompatible with --compress."
        ),
    )
    parser.add_argument(
        "--skip-plsf",
        action="store_true",
        help=(
            "Do not run the pLSF pipeline. "
            "Useful when only the Cubical Ripser baseline is needed, "
            "or when no CUDA device is available."
        ),
    )
    parser.add_argument(
        "--skip-cripser",
        action="store_true",
        help=(
            "Do not run the Cubical Ripser pipeline. "
            "Useful when only the pLSF timing is needed."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Print the raw stdout of each subprocess (plsf and cubicalripser) "
            "in addition to the summary table."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, str]] = []

    # ── pLSF ─────────────────────────────────────────────────────────────
    plsf_filt_ms = None
    if not args.skip_plsf:
        plsf_bin = args.plsf
        if not Path(plsf_bin).exists() and not which(plsf_bin):
            print(f"Error: plsf not found at {plsf_bin}", file=sys.stderr)
            sys.exit(1)

        info = run_plsf_filtration(
            args.input, plsf_bin, args.device, args.verbose,
            compress=args.compress, lossy=args.lossy
        )
        if args.verbose:
            print(info["stdout"])

        rows.append(("File load", "pLSF", fmt_time(info["load_ms"])))
        rows.append(
            ("Filtration construction", "pLSF", fmt_time(info["enum_ms"]))
        )
        rows.append(("Filtration sorting", "pLSF", fmt_time(info["sort_ms"])))
        plsf_filt_ms = info["enum_ms"] + info["sort_ms"]
        rows.append(
            ("Enum + sort total", "pLSF", fmt_time(plsf_filt_ms))
        )

    # ── Cubical Ripser ───────────────────────────────────────────────────
    cr_filt_ms = None
    if not args.skip_cripser:
        cr_bin = args.cubicalripser
        if not Path(cr_bin).exists() and not which(cr_bin):
            print(
                f"Error: cubicalripser not found at {cr_bin}", file=sys.stderr
            )
            sys.exit(1)

        with tempfile.TemporaryDirectory(prefix="cr_bench_") as tmpdir:
            npy_path = Path(tmpdir) / "input.npy"
            conv_ms = nii_to_npy_f64(args.input, npy_path)

            rows.append(("NIfTI -> npy conversion", "CubicalRipser", fmt_time(conv_ms)))

            cr = run_cr_filtration(npy_path, cr_bin, args.maxdim, args.verbose)
            if args.verbose:
                print(cr["stdout"])

            rows.append(("File load (.npy)", "CubicalRipser", fmt_time(cr["load_ms"])))

            for dt in cr["dim_timings"]:
                d = dt["dim"]
                rows.append(
                    (
                        f"Dim-{d} enumeration ({dt['cells']} cells)",
                        "CubicalRipser",
                        fmt_time(dt["enum_ms"]),
                    )
                )
                rows.append(
                    (
                        f"Dim-{d} sorting",
                        "CubicalRipser",
                        fmt_time(dt["sort_ms"]),
                    )
                )

            cr_filt_ms = cr["enum_ms"] + cr["sort_ms"]
            rows.append(
                ("Enum + sort total", "CubicalRipser", fmt_time(cr_filt_ms))
            )

    # ── Summary ──────────────────────────────────────────────────────────
    print_table(rows)

    if plsf_filt_ms is not None and cr_filt_ms is not None:
        if plsf_filt_ms > 0 and cr_filt_ms > 0:
            ratio = cr_filt_ms / plsf_filt_ms
            if ratio >= 1.0:
                faster, factor = "pLSF", ratio
            else:
                faster, factor = "CubicalRipser", 1.0 / ratio
            print(
                f"Speedup: {faster} filtration is {factor:.2f}x faster "
                f"({fmt_time(plsf_filt_ms)} vs {fmt_time(cr_filt_ms)})"
            )


if __name__ == "__main__":
    main()
