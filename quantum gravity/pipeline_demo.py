"""Demonstration pipeline: fetch open data, generate hybrid waveform, inject, whiten."""
from __future__ import annotations

import math

import numpy as np

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from waveform_hybrid import (
    HybridPropagationParams,
    gwpy_to_pycbc,
    hybrid_waveform,
    inject_into_strain,
    whiten_strain,
)


def main():
    event = "GW150914"
    gps = event_gps(event)
    print(f"Fetched GPS for {event}: {gps}")

    # Retrieve 32 s of Hanford strain data
    strain_gwpy: GwpyTimeSeries = GwpyTimeSeries.fetch_open_data("H1", gps - 16, gps + 16)
    strain = gwpy_to_pycbc(strain_gwpy)
    print(f"Strain duration: {strain.duration} s; sample rate: {1/strain.delta_t} Hz")

    params = HybridPropagationParams(
        m_geff_ev=1e-23,
        distance_gpc=0.4,
        foam_strength=1.0,
        rng_seed=2025,
        foam_corr_m=1e-16,
    )
    waveform = hybrid_waveform(36, 29, params=params, f_lower=20.0)

    # Align injection 2 seconds before merger time
    t_ref = gps - 2.0
    injected = inject_into_strain(strain, waveform, t_ref=t_ref, event_start=strain.start_time)

    whitened = whiten_strain(injected)

    vals = whitened.numpy()
    peak_strain = float(np.abs(vals).max())
    rms = float(np.sqrt(np.mean(vals**2)))
    print(f"Whitened peak strain: {peak_strain:.3e}; RMS: {rms:.3e}")


if __name__ == "__main__":
    main()
