"""Matched-filter and inference scaffold prepared for GWTC-4 injections."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import bilby

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
    f_lower: float = 20.0
    psd_fftlength: float = 4.0
    psd_overlap: float = 2.0

    @property
    def distance_mpc(self) -> float:
        return self.distance_gpc * 1e3


PRIMARY_EVENT = EventConfig(
    name="GW230529_181500",  # GWTC-4 high-z target (O4)
    mass1=70.0,
    mass2=55.0,
    distance_gpc=4.8,
    segment=48.0,
    injection_offset=-2.0,
    priors_distance_sigma=300.0,
    detectors=("H1", "L1", "V1"),
    ra=1.85,  # placeholder sky location (radians)
    dec=-0.75,
    psi=0.3,
    theta_jn=1.2,
)

FALLBACK_EVENT = EventConfig(
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
)


def prepare_data(config: EventConfig) -> Tuple[float, Dict[str, GwpyTimeSeries]]:
    gps = event_gps(config.name)
    half = config.segment / 2
    fetched: Dict[str, GwpyTimeSeries] = {}
    for det in config.detectors:
        try:
            fetched[det] = GwpyTimeSeries.fetch_open_data(det, gps - half, gps + half)
        except Exception as exc:  # noqa: BLE001 - propagate context for missing data
            print(f"Failed to fetch {det} data for {config.name}: {exc}")
    missing = sorted(set(config.detectors) - set(fetched))
    if missing:
        raise RuntimeError(f"Missing detectors for {config.name}: {missing}")
    return gps, fetched


def preprocess_data(strains: Dict[str, PycbcTimeSeries], low_freq: float) -> Tuple[Dict[str, PycbcTimeSeries], Dict[str, GwpyTimeSeries]]:
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


def select_event(candidates: Iterable[EventConfig]) -> Tuple[EventConfig, float, Dict[str, GwpyTimeSeries]]:
    last_error: Exception | None = None
    for config in candidates:
        try:
            gps, data = prepare_data(config)
        except Exception as exc:  # noqa: BLE001 - diagnose missing releases
            print(f"Skipping {config.name}: {exc}")
            last_error = exc
            continue
        print(f"Using event {config.name} with detectors {', '.join(data.keys())}")
        return config, gps, data
    raise RuntimeError(f"Failed to retrieve data for configured events: {last_error}")


def compute_detector_snrs(template, data_dict, f_lower):
    peak_stats = {}
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


def run_bilby_stub(
    strain_dict: Dict[str, GwpyTimeSeries],
    gps: float,
    inj_params: HybridPropagationParams,
    config: EventConfig,
    outdir: str,
):
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
        floor = np.min(interp_psd[positive_mask])
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
    print(f"Bilby log-likelihood at injection parameters: {logl:.2f}")

    if os.path.isdir(outdir):
        for fname in os.listdir(outdir):
            os.remove(os.path.join(outdir, fname))

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
        outdir=outdir,
        label="hybrid",
    )
    print(f"Sampler terminated with log_evidence = {result.log_evidence:.2f}")


def main():
    config, gps, raw_strains = select_event((PRIMARY_EVENT, FALLBACK_EVENT))
    pycbc_raw = {det: gwpy_to_pycbc(series) for det, series in raw_strains.items()}

    params = HybridPropagationParams(
        m_geff_ev=1e-23,
        distance_gpc=config.distance_gpc,
        foam_strength=0.8,
        rng_seed=42,
        foam_corr_m=1e-16,
    )
    injection_waveform = hybrid_waveform(
        config.mass1,
        config.mass2,
        params=params,
        distance_mpc=config.distance_mpc,
    )
    t_ref = gps + config.injection_offset
    injected_pycbc = {
        det: inject_into_strain(strain, injection_waveform, t_ref=t_ref, event_start=strain.start_time)
        for det, strain in pycbc_raw.items()
    }

    processed_pycbc, processed_gwpy = preprocess_data(injected_pycbc, config.f_lower)

    reference_detector = next(iter(processed_pycbc))
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=config.mass1,
        mass2=config.mass2,
        delta_t=processed_pycbc[reference_detector].delta_t,
        f_lower=config.f_lower,
    )

    peak_stats, network_snr = compute_detector_snrs(hp, processed_pycbc, config.f_lower)
    for det, (snr_val, snr_time) in peak_stats.items():
        print(f"{det} peak SNR ≈ {snr_val:.2f} at GPS {snr_time}")
    print(f"Network SNR (quadrature) ≈ {network_snr:.2f}")

    run_bilby_stub(processed_gwpy, gps, params, config=config, outdir=f"bilby_stub_{config.name}")


if __name__ == "__main__":
    main()
