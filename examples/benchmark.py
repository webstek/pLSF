#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# benchmark.py – Performance comparison: pLSF + PHAT  vs  Cubical Ripser
# ──────────────────────────────────────────────────────────────────────────────
"""
Benchmark persistent-homology computation on 3D NIfTI volumes.

Pipeline A  (pLSF + PHAT):
    1. plsf  <input.nii>  →  boundary matrix (.bin) + filtration values (.vals)
    2. phat  <stem.bin>   →  persistence pairs

Pipeline B  (Cubical Ripser):
    1. nii2npy  <input.nii>  →  .npy
    2. cripser.computePH(array)  →  persistence pairs

Usage:
    python benchmark.py <input.nii> [options]

Dependencies:
    pip install nibabel numpy cripser tabulate
    External: plsf (this project), phat (https://github.com/xolotl/PHAT)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Utility ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR.parent / "bin"


def which(name: str) -> str | None:
    """Return full path to executable or None."""
    return shutil.which(name)


def fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def print_table(rows: list[tuple[str, str, str]], col_widths: tuple[int, ...] = (36, 14, 14)) -> None:
    """Simple table printer (avoids hard dependency on tabulate)."""
    header = ("Step", "Pipeline", "Time")
    sep = "─"
    print()
    print(f"{'─' * sum(col_widths)}")
    print(f"{header[0]:<{col_widths[0]}}{header[1]:<{col_widths[1]}}{header[2]:>{col_widths[2]}}")
    print(f"{'─' * sum(col_widths)}")
    for label, pipeline, elapsed in rows:
        print(f"{label:<{col_widths[0]}}{pipeline:<{col_widths[1]}}{elapsed:>{col_widths[2]}}")
    print(f"{'─' * sum(col_widths)}")
    print()


# ── Pipeline A: pLSF + PHAT ─────────────────────────────────────────────────

def run_plsf(nii_path: Path, tmpdir: Path, plsf_bin: str, device: str, verbose: bool) -> dict:
    """Run plsf on a NIfTI file; return timing dict."""
    stem = str(tmpdir / "plsf_out")
    cmd = [plsf_bin, str(nii_path), "-o", stem, "-t", "-d", device]
    if verbose:
        cmd.append("-v")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_total = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"plsf failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return {
        "elapsed": elapsed_total,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "bin_path": Path(stem + ".bin"),
        "vals_path": Path(stem + ".vals"),
    }


def run_phat(bin_path: Path, phat_bin: str, algorithm: str) -> dict:
    """Run PHAT on a binary boundary matrix; return timing dict."""
    pairs_path = bin_path.with_suffix(".pairs")
    cmd = [phat_bin, "--binary", str(bin_path), str(pairs_path)]
    if algorithm:
        cmd.extend(["--algorithm", algorithm])

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"phat failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return {"elapsed": elapsed, "stdout": result.stdout, "stderr": result.stderr}


# ── Pipeline B: Cubical Ripser ───────────────────────────────────────────────

def run_cubical_ripser(nii_path: Path, verbose: bool) -> dict:
    """Load NIfTI as numpy, run cripser.computePH, return timing dict."""
    try:
        import cripser
    except ImportError:
        print(
            "Error: cripser not installed.  pip install cripser",
            file=sys.stderr,
        )
        sys.exit(1)

    import nibabel as nib
    import numpy as np

    # Load volume
    t0 = time.perf_counter()
    img = nib.load(str(nii_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    load_time = time.perf_counter() - t0

    if verbose:
        print(f"Cubical Ripser input: shape={data.shape}  dtype={data.dtype}")

    # Compute PH
    t1 = time.perf_counter()
    ph = cripser.computePH(data)
    compute_time = time.perf_counter() - t1

    return {
        "load_time": load_time,
        "compute_time": compute_time,
        "total": load_time + compute_time,
        "num_pairs": len(ph),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pLSF+PHAT vs Cubical Ripser on 3D NIfTI data."
    )
    parser.add_argument("input", type=Path, help="Input .nii file.")
    parser.add_argument(
        "--plsf",
        default=str(BIN_DIR / "plsf"),
        help="Path to plsf binary (default: ../bin/plsf relative to script).",
    )
    parser.add_argument(
        "--phat",
        default="phat",
        help="Path to PHAT binary (default: phat in PATH).",
    )
    parser.add_argument(
        "--phat-algorithm",
        default="",
        help="PHAT reduction algorithm (e.g. chunk_reduction, spectral_sequence).",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu", "default"],
        help="SYCL device for plsf (default: gpu).",
    )
    parser.add_argument(
        "--skip-plsf", action="store_true", help="Skip the pLSF+PHAT pipeline."
    )
    parser.add_argument(
        "--skip-cripser",
        action="store_true",
        help="Skip the Cubical Ripser pipeline.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, str]] = []

    # ── Pipeline A ───────────────────────────────────────────────────────
    plsf_total = None
    if not args.skip_plsf:
        plsf_bin = args.plsf
        if not Path(plsf_bin).exists() and not which(plsf_bin):
            print(f"Error: plsf not found at {plsf_bin}", file=sys.stderr)
            sys.exit(1)

        phat_bin = args.phat
        if not Path(phat_bin).exists() and not which(phat_bin):
            print(
                f"Warning: phat not found at '{phat_bin}'; "
                "only plsf timing will be reported.",
                file=sys.stderr,
            )
            phat_bin = None

        with tempfile.TemporaryDirectory(prefix="plsf_bench_") as tmpdir:
            tmpdir = Path(tmpdir)

            # plsf
            info = run_plsf(
                args.input, tmpdir, plsf_bin, args.device, args.verbose
            )
            rows.append(("Lower-star filtration + boundary", "pLSF+PHAT", fmt_time(info["elapsed"])))
            if args.verbose:
                print(info["stdout"])

            # phat
            if phat_bin and info["bin_path"].exists():
                phat_info = run_phat(info["bin_path"], phat_bin, args.phat_algorithm)
                rows.append(("Reduction (persistent homology)", "pLSF+PHAT", fmt_time(phat_info["elapsed"])))
                plsf_total = info["elapsed"] + phat_info["elapsed"]
            else:
                plsf_total = info["elapsed"]

            rows.append(("Pipeline total", "pLSF+PHAT", fmt_time(plsf_total)))

    # ── Pipeline B ───────────────────────────────────────────────────────
    cr_total = None
    if not args.skip_cripser:
        cr = run_cubical_ripser(args.input, args.verbose)
        rows.append(("Volume load (nibabel)", "CubicalRipser", fmt_time(cr["load_time"])))
        rows.append(("Filtration + reduction", "CubicalRipser", fmt_time(cr["compute_time"])))
        rows.append(("Pipeline total", "CubicalRipser", fmt_time(cr["total"])))
        cr_total = cr["total"]

        if args.verbose:
            print(f"Cubical Ripser pairs: {cr['num_pairs']}")

    # ── Summary ──────────────────────────────────────────────────────────
    print_table(rows)

    if plsf_total is not None and cr_total is not None:
        ratio = cr_total / plsf_total if plsf_total > 0 else float("inf")
        faster = "pLSF+PHAT" if plsf_total < cr_total else "CubicalRipser"
        print(
            f"Speedup: {faster} is {max(ratio, 1/ratio):.2f}x faster "
            f"({fmt_time(plsf_total)} vs {fmt_time(cr_total)})"
        )


if __name__ == "__main__":
    main()
