"""Parameter sensitivity exploration for hybrid waveform perturbations."""
from __future__ import annotations

import itertools
import numpy as np

from phase_shift_estimates import phase_shift
from waveform_hybrid import HybridPropagationParams, hybrid_waveform, C, L_PL, GPC


def waveform_metrics(gr_waveform, perturbed_waveform):
    diff = perturbed_waveform - gr_waveform
    l2 = float(np.linalg.norm(diff.numpy()))
    max_abs = float(np.max(np.abs(diff.numpy())))
    rel_norm = l2 / float(np.linalg.norm(gr_waveform.numpy()))
    return l2, max_abs, rel_norm


def scan_grid(mass1, mass2, m_values, foam_values, distance_gpc=0.4, seed=0, foam_corr_m=1e-16):
    baseline = hybrid_waveform(mass1, mass2, params=HybridPropagationParams(
        m_geff_ev=0.0,
        distance_gpc=distance_gpc,
        foam_strength=0.0,
        foam_corr_m=foam_corr_m,
    ), distance_mpc=distance_gpc * 1e3)
    results = []
    for m_geff, foam in itertools.product(m_values, foam_values):
        params = HybridPropagationParams(
            m_geff_ev=m_geff,
            distance_gpc=distance_gpc,
            foam_strength=foam,
            rng_seed=seed,
            foam_corr_m=foam_corr_m,
        )
        perturbed = hybrid_waveform(mass1, mass2, params=params, distance_mpc=distance_gpc * 1e3)
        l2, max_abs, rel_norm = waveform_metrics(baseline, perturbed)
        delta_phi = phase_shift(m_geff, distance_gpc=distance_gpc, frequency_hz=100.0)
        foam_rms = foam * np.sqrt(100.0 * distance_gpc * GPC * (L_PL**3) / C)
        results.append(
            {
                "m_geff": m_geff,
                "foam_strength": foam,
                "l2_norm": l2,
                "max_abs": max_abs,
                "relative_norm": rel_norm,
                "phase_shift": delta_phi,
                "foam_rms": foam_rms,
            }
        )
    return results


def main():
    m_values = [0.0, 1e-23, 1e-21, 1e-19, 1e-18]
    foam_values = [0.0, 1.0, 5.0, 10.0]
    results = scan_grid(36, 29, m_values, foam_values)
    print("m_geff_eV,foam_strength,l2_norm,max_abs,relative_norm,phase_shift,foam_rms")
    for row in results:
        print(
            f"{row['m_geff']:.1e},{row['foam_strength']:.2f},"
            f"{row['l2_norm']:.3e},{row['max_abs']:.3e},{row['relative_norm']:.3e},"
            f"{row['phase_shift']:.3e},{row['foam_rms']:.3e}"
        )


if __name__ == "__main__":
    main()
