"""Explore waveform sensitivity versus foam coherence length."""
from __future__ import annotations

import numpy as np

from waveform_hybrid import HybridPropagationParams, hybrid_waveform


def coherence_scan(
    mass1: float,
    mass2: float,
    distance_gpc: float,
    m_geff_ev: float,
    foam_strength: float,
    lengths: np.ndarray,
    seed: int = 1,
):
    baseline = hybrid_waveform(
        mass1,
        mass2,
        params=HybridPropagationParams(
            m_geff_ev=m_geff_ev,
            distance_gpc=distance_gpc,
            foam_strength=0.0,
            foam_corr_m=None,
            rng_seed=seed,
        ),
        distance_mpc=distance_gpc * 1e3,
    )
    results = []
    base_norm = float(np.linalg.norm(baseline.numpy()))
    for length in lengths:
        params = HybridPropagationParams(
            m_geff_ev=m_geff_ev,
            distance_gpc=distance_gpc,
            foam_strength=foam_strength,
            foam_corr_m=float(length),
            rng_seed=seed,
        )
        perturbed = hybrid_waveform(
            mass1,
            mass2,
            params=params,
            distance_mpc=distance_gpc * 1e3,
        )
        diff = perturbed - baseline
        l2 = float(np.linalg.norm(diff.numpy()))
        rel = l2 / base_norm if base_norm else 0.0
        results.append((length, l2, rel, diff.numpy().std()))
    return results


def main():
    lengths = np.logspace(-20, -14, 13)
    data = coherence_scan(
        mass1=70.0,
        mass2=55.0,
        distance_gpc=4.8,
        m_geff_ev=1e-23,
        foam_strength=0.8,
        lengths=lengths,
    )
    print("foam_corr_m[L], l2_norm, rel_norm, diff_std")
    for length, l2, rel, std in data:
        print(f"{length:.2e}, {l2:.3e}, {rel:.3e}, {std:.3e}")


if __name__ == "__main__":
    main()
