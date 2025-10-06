"""Summarize posterior statistics from a saved bilby result."""
from __future__ import annotations

import argparse
from pathlib import Path

import bilby
import numpy as np

DEFAULT_PARAMS = ("m_geff_ev", "foam_strength", "foam_corr_m")


def quantile_summary(samples: np.ndarray, probs: tuple[float, ...]) -> list[float]:
    return [float(np.quantile(samples, p)) for p in probs]


def summarize(result_path: Path, parameters: tuple[str, ...], credible: float) -> None:
    result = bilby.result.read_in_result(filename=str(result_path))
    posterior = result.posterior

    probs = ((1 - credible) / 2, 0.5, 1 - (1 - credible) / 2)
    print(f"Result: {result.label} (logZ = {result.log_evidence:.2f} ± {result.log_evidence_err:.2f})")
    print(f"Noise logZ = {result.log_noise_evidence:.2f}; logB = {result.log_bayes_factor:.2f}")
    print(f"Credible interval level: {credible:.3f}")

    for param in parameters:
        if param not in posterior:
            print(f"- {param}: not in posterior")
            continue
        mean = float(posterior[param].mean())
        std = float(posterior[param].std())
        q_low, q_med, q_high = quantile_summary(posterior[param].to_numpy(), probs)
        print(
            f"- {param}: mean={mean:.4e}, std={std:.4e}, "
            f"[{q_low:.4e}, {q_med:.4e}, {q_high:.4e}] ({credible*100:.1f}% CI)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        type=Path,
        help="Path to bilby_result.json",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        default=DEFAULT_PARAMS,
        help="Posterior parameters to summarize",
    )
    parser.add_argument(
        "--credible",
        type=float,
        default=0.9,
        help="Credible interval level (default 0.9)",
    )
    args = parser.parse_args()

    summarize(args.result, tuple(args.params), args.credible)


if __name__ == "__main__":
    main()
