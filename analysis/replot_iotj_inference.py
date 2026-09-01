#!/usr/bin/env python3
"""Replot the IoTJ inference/decision figure from committed seed outputs."""

from __future__ import annotations

import pandas as pd

import iotj_inference_evidence as evidence


def main() -> None:
    stationary = pd.read_csv(
        evidence.RESULTS / "stationary_prediction_summary.csv"
    )
    shift_windows = pd.read_csv(
        evidence.RESULTS / "shift_prediction_windows.csv"
    )
    shift_summary = pd.read_csv(
        evidence.RESULTS / "shift_prediction_summary.csv"
    )
    evidence.render_figure(stationary, shift_windows, shift_windows)
    evidence.write_summary(stationary, shift_summary)
    print("rebuilt figures/fig_iotj_inference.{pdf,png,svg}")


if __name__ == "__main__":
    main()
