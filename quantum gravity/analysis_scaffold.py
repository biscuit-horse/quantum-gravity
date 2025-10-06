"""Matched-filter and inference scaffold for hybrid quantum-foam graviton models."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import bilby
import numpy as np
import shutil

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from pycbc.filter import matched_filter
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types.timeseries import TimeSeries as PycbcTimeSeries
from pycbc.waveform import get_td_waveform

from waveform_hybrid import (
    HybridPropagationParams,
    gwpy_to_pycbc,
    pycbc_to_gwpy,
    hybrid_waveform,
    bilby_time_domain_waveform,
    inject_into_strain,
)


@dataclass(frozen=True)
class EventConfig:
    name: str
    mass1: float
    mass2: float
    distance_gpc: float
    segment: float
    injection_offset: float
    priors_distance_sigma: float
    detectors: Tuple[str, ...]
    ra: float
    dec: float
    psi: float
    theta_jn: float
    m_geff_ev: float = 1e-23
    foam_strength: float = 0.8
    foam_corr_m: float = 1e-16
    rng_seed: int = 42
    f_lower: float = 20.0
    psd_fftlength: float = 4.0
    psd_overlap: float = 2.0

    @property
    def distance_mpc(self) -> float:
        return self.distance_gpc * 1e3


EVENTS: Dict[str, EventConfig] = {
    cfg.name: cfg
    for cfg in (
        EventConfig(
            name="GW230529_181500",  # GWTC-4 target (data currently unavailable, handled gracefully)
            mass1=70.0,
            mass2=55.0,
            distance_gpc=4.8,
            segment=48.0,
            injection_offset=-2.0,
            priors_distance_sigma=300.0,
            detectors=("H1", "L1", "V1"),
            ra=1.85,
            dec=-0.75,
            psi=0.3,
            theta_jn=1.2,
            m_geff_ev=1e-23,
            foam_strength=0.8,
            foam_corr_m=1e-16,
            rng_seed=101,
        ),
        EventConfig(
            name="GW190521",
            mass1=85.0,
            mass2=66.0,
            distance_gpc=5.3,
            segment=32.0,
            injection_offset=-2.0,
            priors_distance_sigma=500.0,
            detectors=("H1", "L1"),
            ra=1.0,
            dec=-1.0,
            psi=0.0,
            theta_jn=1.0,
            m_geff_ev=1e-23,
            foam_strength=0.8,
            foam_corr_m=1e-16,
            rng_seed=42,
        ),
        EventConfig(
            name="GW150914",
            mass1=36.0,
            mass2=29.0,
            distance_gpc=0.41,
            segment=32.0,
            injection_offset=-2.0,
            priors_distance_sigma=80.0,
            detectors=("H1", "L1"),
            ra=2.10,
            dec=-1.24,
            psi=0.0,
            theta_jn=1.3,
            m_geff_ev=1e-23,
            foam_strength=0.8,
            foam_corr_m=5e-17,
            rng_seed=84,
        ),
    )
}


def prepare_data(config: EventConfig) -> Tuple[float, Dict[str, GwpyTimeSeries]]:
    gps = event_gps(config.name)
    half = config.segment / 2
    fetched: Dict[str, GwpyTimeSeries] = {}
    for det in config.detectors:
        try:
            fetched[det] = GwpyTimeSeries.fetch_open_data(det, gps - half, gps + half)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to fetch {det} data for {config.name}: {exc}")
    missing = sorted(set(config.detectors) - set(fetched))
    if missing:
        raise RuntimeError(f"Missing detectors for {config.name}: {missing}")
    return gps, fetched


def preprocess_data(
    strains: Dict[str, PycbcTimeSeries],
    low_freq: float,
) -> Tuple[Dict[str, PycbcTimeSeries], Dict[str, GwpyTimeSeries]]:
    processed_pycbc: Dict[str, PycbcTimeSeries] = {}
    processed_gwpy: Dict[str, GwpyTimeSeries] = {}
    for det, series in strains.items():
        filtered = series.highpass_fir(low_freq, 256)
        cropped = filtered.crop(4, 4)
        processed_pycbc[det] = cropped
        processed_gwpy[det] = pycbc_to_gwpy(cropped)
    return processed_pycbc, processed_gwpy


def resize_template(template, target_length):
    template = template.copy()
    template.resize(target_length)
    template.start_time = 0
    return template


def compute_detector_snrs(template, data_dict, f_lower):
    peak_stats: Dict[str, Tuple[float, float]] = {}
    network_snr_sq = 0.0
    for det, strain in data_dict.items():
        template_resized = resize_template(template, len(strain))
        psd = aLIGOZeroDetHighPower(len(strain) // 2 + 1, strain.delta_f, f_lower)
        snr = matched_filter(template_resized, strain, psd=psd, low_frequency_cutoff=f_lower)
        snr = snr.crop(4, 4)
        peak_idx = int(np.argmax(np.abs(snr.numpy())))
        peak_time = float(snr.sample_times[peak_idx])
        peak_val = float(np.abs(snr.numpy()[peak_idx]))
        peak_stats[det] = (peak_val, peak_time)
        network_snr_sq += peak_val**2
    network_snr = math.sqrt(network_snr_sq)
    return peak_stats, network_snr


def build_params(event: EventConfig, model: str, seed: int) -> HybridPropagationParams:
    if model == "gr":
        return HybridPropagationParams(
            m_geff_ev=0.0,
            distance_gpc=event.distance_gpc,
            foam_strength=0.0,
            rng_seed=seed,
            foam_corr_m=event.foam_corr_m,
        )
    if model == "hybrid":
        return HybridPropagationParams(
            m_geff_ev=event.m_geff_ev,
            distance_gpc=event.distance_gpc,
            foam_strength=event.foam_strength,
            rng_seed=seed,
            foam_corr_m=event.foam_corr_m,
        )
    raise ValueError(f"Unknown model '{model}'")


def run_bilby_stub(
    strain_dict: Dict[str, GwpyTimeSeries],
    gps: float,
    inj_params: HybridPropagationParams,
    config: EventConfig,
    outdir: Path,
    model: str,
    label: str,
) -> bilby.result.Result:
    duration = next(iter(strain_dict.values())).duration
    sampling_frequency = 1 / next(iter(strain_dict.values())).dt.value

    interferometers: List[bilby.gw.detector.Interferometer] = []
    for det, series in strain_dict.items():
        interferometer = bilby.gw.detector.get_empty_interferometer(det)
        interferometer.minimum_frequency = config.f_lower
        interferometer.maximum_frequency = sampling_frequency / 2
        interferometer.set_strain_data_from_gwpy_timeseries(series)

        psd_series = series.psd(
            fftlength=config.psd_fftlength,
            overlap=config.psd_overlap,
            window="hann",
        )
        psd_freq = psd_series.frequencies.value
        psd_vals = psd_series.value.real
        target_freq = interferometer.frequency_array
        interp_psd = np.interp(target_freq, psd_freq, psd_vals, left=psd_vals[0], right=psd_vals[-1])
        positive_mask = interp_psd > 0
        if not np.any(positive_mask):
            raise RuntimeError(f"PSD estimation failed for {det}")
        floor = float(np.min(interp_psd[positive_mask]))
        interp_psd[~positive_mask] = floor
        interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=target_freq,
            psd_array=interp_psd,
        )
        interferometers.append(interferometer)

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomD",
        f_lower=config.f_lower,
    )
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        time_domain_source_model=bilby_time_domain_waveform,
        waveform_arguments=waveform_arguments,
    )

    priors = bilby.core.prior.PriorDict()
    priors["mass_1"] = bilby.core.prior.Uniform(config.mass1 * 0.7, config.mass1 * 1.2, "mass_1")
    priors["mass_2"] = bilby.core.prior.Uniform(config.mass2 * 0.7, config.mass2 * 1.2, "mass_2")
    priors["luminosity_distance"] = bilby.core.prior.TruncatedGaussian(
        mu=config.distance_mpc,
        sigma=config.priors_distance_sigma,
        minimum=max(100.0, config.distance_mpc - 5 * config.priors_distance_sigma),
        maximum=config.distance_mpc + 5 * config.priors_distance_sigma,
        name="luminosity_distance",
    )
    priors["theta_jn"] = bilby.core.prior.Sine(name="theta_jn")
    priors["phase"] = bilby.core.prior.Uniform(0.0, 2 * math.pi, "phase")
    priors["psi"] = bilby.core.prior.DeltaFunction(config.psi)
    priors["ra"] = bilby.core.prior.DeltaFunction(config.ra)
    priors["dec"] = bilby.core.prior.DeltaFunction(config.dec)
    priors["geocent_time"] = bilby.core.prior.Uniform(gps - 0.005, gps + 0.005, "geocent_time")

    if model == "gr":
        priors["m_geff_ev"] = bilby.core.prior.DeltaFunction(0.0)
        priors["foam_strength"] = bilby.core.prior.DeltaFunction(0.0)
        priors["foam_corr_m"] = bilby.core.prior.DeltaFunction(max(inj_params.foam_corr_m, 1e-20))
    else:
        priors["m_geff_ev"] = bilby.core.prior.LogUniform(1e-26, 1e-22, "m_geff_ev")
        priors["foam_strength"] = bilby.core.prior.Uniform(0.0, 1.5, "foam_strength")
        priors["foam_corr_m"] = bilby.core.prior.LogUniform(1e-20, 1e-14, "foam_corr_m")

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        time_marginalization=False,
        distance_marginalization=False,
        phase_marginalization=False,
    )

    injection_parameters = dict(
        mass_1=config.mass1,
        mass_2=config.mass2,
        luminosity_distance=config.distance_mpc,
        theta_jn=config.theta_jn,
        phase=0.0,
        psi=config.psi,
        ra=config.ra,
        dec=config.dec,
        geocent_time=gps,
        m_geff_ev=inj_params.m_geff_ev,
        foam_strength=inj_params.foam_strength,
        foam_corr_m=inj_params.foam_corr_m,
    )
    likelihood.parameters.update(injection_parameters)
    logl = likelihood.log_likelihood()
    print(f"Bilby log-likelihood at injection parameters ({model}) = {logl:.2f}")

    # Clear previous contents for reproducibility
    if outdir.exists():
        for item in outdir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=32,
        dlogz=2.0,
        maxmcmc=32,
        walks=12,
        nact=5,
        npool=1,
        clean=False,
        outdir=str(outdir),
        label=label,
    )
    print(f"Sampler terminated with log_evidence = {result.log_evidence:.2f}")
    return result


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        nargs="*",
        default=["GW230529_181500", "GW190521", "GW150914"],
        help="Event names to attempt, in priority order.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["hybrid", "gr"],
        default=["hybrid", "gr"],
        help="Models to evaluate per event/seed.",
    )
    parser.add_argument(
        "--injection-model",
        choices=["hybrid", "gr"],
        default="hybrid",
        help="Model used when injecting signals into the strain.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of Monte Carlo jitter seeds per event (>=1).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--skip-bilby",
        action="store_true",
        help="If set, skip Bilby inference and only report SNRs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_events = [evt for evt in args.events if evt in EVENTS]
    if not requested_events:
        raise SystemExit("No recognised events requested.")

    args.output_root.mkdir(parents=True, exist_ok=True)

    for event_name in requested_events:
        event = EVENTS[event_name]
        try:
            gps, raw_strains = prepare_data(event)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {event_name}: {exc}")
            continue

        pycbc_raw = {det: gwpy_to_pycbc(series) for det, series in raw_strains.items()}

        for seed_index in range(args.num_seeds):
            seed = event.rng_seed + seed_index
            injection_params = build_params(event, args.injection_model, seed)
            injection_waveform = hybrid_waveform(
                event.mass1,
                event.mass2,
                params=injection_params,
                distance_mpc=event.distance_mpc,
            )
            t_ref = gps + event.injection_offset
            injected_pycbc = {
                det: inject_into_strain(strain, injection_waveform, t_ref=t_ref, event_start=strain.start_time)
                for det, strain in pycbc_raw.items()
            }

            processed_pycbc, processed_gwpy = preprocess_data(injected_pycbc, event.f_lower)

            reference_detector = next(iter(processed_pycbc))
            hp, _ = get_td_waveform(
                approximant="IMRPhenomD",
                mass1=event.mass1,
                mass2=event.mass2,
                delta_t=processed_pycbc[reference_detector].delta_t,
                f_lower=event.f_lower,
            )
            peak_stats, network_snr = compute_detector_snrs(hp, processed_pycbc, event.f_lower)

            snr_summary = {
                det: {"snr": float(val), "gps": float(time)} for det, (val, time) in peak_stats.items()
            }
            snr_summary["network"] = network_snr

            for model in args.models:
                model_params = build_params(event, model, seed)
                run_dir = args.output_root / event.name / model / f"seed_{seed:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                result = None
                if not args.skip_bilby:
                    result = run_bilby_stub(
                        processed_gwpy,
                        gps,
                        model_params,
                        config=event,
                        outdir=run_dir,
                        model=model,
                        label=f"{model}_seed{seed:03d}",
                    )

                summary = {
                    "event": event.name,
                    "model": model,
                    "injection_model": args.injection_model,
                    "seed": seed,
                    "detectors": sorted(processed_pycbc.keys()),
                    "snr": snr_summary,
                    "gps": gps,
                    "injection_params": {
                        "m_geff_ev": injection_params.m_geff_ev,
                        "foam_strength": injection_params.foam_strength,
                        "foam_corr_m": injection_params.foam_corr_m,
                    },
                    "model_params": {
                        "m_geff_ev": model_params.m_geff_ev,
                        "foam_strength": model_params.foam_strength,
                        "foam_corr_m": model_params.foam_corr_m,
                    },
                }

                if result is not None:
                    summary["log_evidence"] = float(result.log_evidence)
                    summary["log_evidence_err"] = float(result.log_evidence_err)
                    summary["log_noise_evidence"] = float(result.log_noise_evidence)
                    summary["log_bayes_factor"] = float(result.log_bayes_factor)

                write_summary(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
