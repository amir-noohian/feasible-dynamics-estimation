#!/usr/bin/env python3
"""
Compare two whitespace-separated .dat files.

Expected format:
    time  signal_1  signal_2 ... signal_8

The files may have different sampling rates. File 2 is interpolated onto
File 1's time samples over their common time interval.

Usage:
    python3 compare_dat.py file1.dat file2.dat

Optional:
    python3 compare_dat.py file1.dat file2.dat --labels q1 q2 q3 q4 tau1 tau2 tau3 tau4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(path: Path) -> np.ndarray:
    """Load and validate a whitespace-separated numeric data file."""
    try:
        data = np.loadtxt(path)
    except OSError as exc:
        raise SystemExit(f"Could not open {path}: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"Could not parse numeric data in {path}: {exc}") from exc

    if data.ndim != 2:
        raise SystemExit(f"{path} must contain a 2-D numeric table.")

    if data.shape[1] < 2:
        raise SystemExit(f"{path} must contain a timestamp and at least one signal.")

    if not np.all(np.isfinite(data)):
        raise SystemExit(f"{path} contains NaN or infinite values.")

    if np.any(np.diff(data[:, 0]) <= 0):
        raise SystemExit(f"Timestamps in {path} must be strictly increasing.")

    return data


def estimate_rate(time: np.ndarray) -> float:
    """Estimate sampling frequency using the median sample interval."""
    return 1.0 / np.median(np.diff(time))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two numeric .dat files.")
    parser.add_argument("file1", type=Path)
    parser.add_argument("file2", type=Path)
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional signal labels. Supply one label per signal column.",
    )
    parser.add_argument(
        "--output",
        default="comparison.png",
        help="Output plot filename. Default: comparison.png",
    )
    args = parser.parse_args()

    data1 = load_data(args.file1)
    data2 = load_data(args.file2)

    if data1.shape[1] != data2.shape[1]:
        raise SystemExit(
            f"Column mismatch: {args.file1} has {data1.shape[1]} columns, "
            f"while {args.file2} has {data2.shape[1]} columns."
        )

    number_of_signals = data1.shape[1] - 1

    if args.labels:
        if len(args.labels) != number_of_signals:
            raise SystemExit(
                f"Expected {number_of_signals} labels, but received "
                f"{len(args.labels)}."
            )
        labels = args.labels
    else:
        labels = [f"Signal {i + 1}" for i in range(number_of_signals)]

    # Convert absolute timestamps to elapsed time.
    time1 = data1[:, 0] - data1[0, 0]
    time2 = data2[:, 0] - data2[0, 0]

    signals1 = data1[:, 1:]
    signals2 = data2[:, 1:]

    # Compare only the interval available in both recordings.
    common_start = max(time1[0], time2[0])
    common_end = min(time1[-1], time2[-1])

    mask1 = (time1 >= common_start) & (time1 <= common_end)
    comparison_time = time1[mask1]
    compared_signals1 = signals1[mask1]

    # Interpolate file 2 onto file 1's timestamps.
    compared_signals2 = np.column_stack(
        [
            np.interp(comparison_time, time2, signals2[:, column])
            for column in range(number_of_signals)
        ]
    )

    error = compared_signals1 - compared_signals2

    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    max_abs_error = np.max(np.abs(error), axis=0)

    # Normalized RMSE using the range of file 1.
    ranges = np.ptp(compared_signals1, axis=0)
    nrmse_percent = np.divide(
        rmse,
        ranges,
        out=np.full_like(rmse, np.nan),
        where=ranges > 0,
    ) * 100.0

    print(f"\nFile 1: {args.file1}")
    print(f"  Samples: {len(data1)}")
    print(f"  Estimated sampling rate: {estimate_rate(time1):.2f} Hz")
    print(f"  Duration: {time1[-1]:.3f} s")

    print(f"\nFile 2: {args.file2}")
    print(f"  Samples: {len(data2)}")
    print(f"  Estimated sampling rate: {estimate_rate(time2):.2f} Hz")
    print(f"  Duration: {time2[-1]:.3f} s")

    print("\nComparison metrics after time alignment")
    print(
        f"{'Signal':<16}"
        f"{'RMSE':>14}"
        f"{'MAE':>14}"
        f"{'Max |error|':>16}"
        f"{'NRMSE (%)':>14}"
    )
    print("-" * 74)

    for index, label in enumerate(labels):
        nrmse_text = (
            f"{nrmse_percent[index]:.6f}"
            if np.isfinite(nrmse_percent[index])
            else "undefined"
        )
        print(
            f"{label:<16}"
            f"{rmse[index]:>14.6f}"
            f"{mae[index]:>14.6f}"
            f"{max_abs_error[index]:>16.6f}"
            f"{nrmse_text:>14}"
        )

    # Plot each signal and its difference.
    figure, axes = plt.subplots(
        number_of_signals,
        2,
        figsize=(15, 2.6 * number_of_signals),
        sharex=True,
        squeeze=False,
    )

    file1_name = args.file1.name
    file2_name = args.file2.name

    for index, label in enumerate(labels):
        axes[index, 0].plot(
            comparison_time,
            compared_signals1[:, index],
            label=file1_name,
            linewidth=1.2,
        )
        axes[index, 0].plot(
            comparison_time,
            compared_signals2[:, index],
            label=file2_name,
            linewidth=1.2,
            alpha=0.8,
        )
        axes[index, 0].set_ylabel(label)
        axes[index, 0].grid(True)
        axes[index, 0].legend(loc="best")

        axes[index, 1].plot(
            comparison_time,
            error[:, index],
            linewidth=1.0,
        )
        axes[index, 1].axhline(0.0, linewidth=0.8)
        axes[index, 1].set_ylabel(f"{label} error")
        axes[index, 1].grid(True)
        axes[index, 1].set_title(
            f"RMSE = {rmse[index]:.5g}, "
            f"NRMSE = {nrmse_percent[index]:.3g}%"
        )

    axes[-1, 0].set_xlabel("Elapsed time [s]")
    axes[-1, 1].set_xlabel("Elapsed time [s]")
    axes[0, 0].set_title("Aligned signals")
    axes[0, 1].set_title("File 1 - File 2")

    figure.tight_layout()
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to: {args.output}")

    plt.show()


if __name__ == "__main__":
    main()
