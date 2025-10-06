"""Hybrid waveform utilities with foam-induced graviton mass corrections."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform

HBAR = 1.054571817e-34  # J*s
C = 2.99792458e8  # m/s
EV_TO_J = 1.602176634e-19  # J/eV
L_PL = 1.616255e-35  # m
PARSEC = 3.0856775814913673e16  # m
GPC = 1e9 * PARSEC


@dataclass
class HybridPropagationParams:
    m_geff_ev: float  # effective graviton mass (eV)
    distance_gpc: float  # luminosity distance (Gpc)
    foam_strength: float  # dimensionless scaling of foam variance (order unity)
    rng_seed: Optional[int] = None
    foam_corr_m: Optional[float] = None  # effective foam coherence length (m)

    @property
    def distance_m(self) -> float:
        return self.distance_gpc * GPC

    @property
    def m_geff_kg(self) -> float:
        return (self.m_geff_ev * EV_TO_J) / (C**2)

    @property
    def corr_length_m(self) -> float:
        return self.foam_corr_m if self.foam_corr_m is not None else L_PL


def apply_mass_phase(freq_series, params: HybridPropagationParams):
    """Apply mass-induced phase shift to a PyCBC FrequencySeries."""
    freqs = freq_series.sample_frequencies.numpy()
    omega = 2 * math.pi * freqs
    phase_shift = np.zeros_like(freqs)
    nonzero = freqs > 0
    phase_shift[nonzero] = (
        (params.m_geff_kg**2) * (C**4) * params.distance_m
        / (2 * HBAR * omega[nonzero])
    )
    freq_series.data *= np.exp(1j * phase_shift)
    return freq_series


def apply_foam_jitter(freq_series, params: HybridPropagationParams):
    """Apply stochastic foam amplitude fluctuations."""
    rng = np.random.default_rng(params.rng_seed)
    freqs = freq_series.sample_frequencies.numpy()
    length_scale = params.corr_length_m
    enhanced = params.foam_strength * np.sqrt(
        np.clip(freqs, 0, None) * params.distance_m * (length_scale**3) / C
    )
    noise_real = rng.normal(scale=enhanced)
    noise_imag = rng.normal(scale=enhanced)
    jitter = noise_real + 1j * noise_imag
    freq_series.data *= (1 + jitter)
    return freq_series


def hybrid_waveform(
    mass1,
    mass2,
    delta_t=1 / 4096,
    f_lower=20.0,
    params: Optional[HybridPropagationParams] = None,
    distance_mpc: Optional[float] = None,
    inclination: Optional[float] = None,
    coa_phase: Optional[float] = None,
    approximant: str = "IMRPhenomD",
    return_cross: bool = False,
):
    wf_kwargs = dict(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        delta_t=delta_t,
        f_lower=f_lower,
    )
    if distance_mpc is not None:
        wf_kwargs["distance"] = distance_mpc
    if inclination is not None:
        wf_kwargs["inclination"] = inclination
    if coa_phase is not None:
        wf_kwargs["coa_phase"] = coa_phase

    hp, hc = get_td_waveform(**wf_kwargs)

    freq_hp = hp.to_frequencyseries()
    freq_hc = hc.to_frequencyseries()
    if params is not None:
        freq_hp = apply_mass_phase(freq_hp, params)
        freq_hc = apply_mass_phase(freq_hc, params)
        freq_hp = apply_foam_jitter(freq_hp, params)
        jitter_params = params
        if params.rng_seed is not None:
            jitter_params = replace(params, rng_seed=params.rng_seed + 1)
        freq_hc = apply_foam_jitter(freq_hc, jitter_params)
    hp_modified = freq_hp.to_timeseries()
    hc_modified = freq_hc.to_timeseries()
    hp_modified.start_time = hp.start_time
    hc_modified.start_time = hc.start_time
    if return_cross:
        return hp_modified, hc_modified
    return hp_modified


def hybrid_ensemble(
    mass1: float,
    mass2: float,
    params: HybridPropagationParams,
    n_realizations: int,
    delta_t: float = 1 / 4096,
    f_lower: float = 20.0,
) -> List[TimeSeries]:
    """Generate an ensemble of stochastic hybrid waveforms."""
    waveforms: List[TimeSeries] = []
    base_seed = params.rng_seed or 0
    for idx in range(n_realizations):
        seed = base_seed + idx
        realization_params = HybridPropagationParams(
            m_geff_ev=params.m_geff_ev,
            distance_gpc=params.distance_gpc,
            foam_strength=params.foam_strength,
            rng_seed=seed,
            foam_corr_m=params.foam_corr_m,
        )
        waveforms.append(
            hybrid_waveform(
                mass1,
                mass2,
                delta_t=delta_t,
                f_lower=f_lower,
                params=realization_params,
            )
        )
    return waveforms


def _ensure_sampling(series: TimeSeries, target_delta_t: float) -> TimeSeries:
    if not math.isclose(float(series.delta_t), target_delta_t, rel_tol=1e-9):
        resampled = series.copy()
        resampled = resampled.resample(1 / target_delta_t)
        return resampled
    return series


def _match_length(series: TimeSeries, target_len: int) -> np.ndarray:
    data = series.numpy()
    if len(data) == target_len:
        return data
    if len(data) > target_len:
        return data[-target_len:]
    padded = np.zeros(target_len, dtype=data.dtype)
    padded[-len(data) :] = data
    return padded


def bilby_time_domain_waveform(
    time_array,
    mass_1,
    mass_2,
    luminosity_distance,
    m_geff_ev,
    foam_strength,
    foam_corr_m=None,
    theta_jn=0.0,
    phase=0.0,
    **kwargs,
):
    """Return plus/cross arrays on bilby's time grid with hybrid corrections."""
    f_lower = kwargs.get("f_lower", 20.0)
    approximant = kwargs.get("waveform_approximant", "IMRPhenomD")

    delta_t = float(time_array[1] - time_array[0])
    params = HybridPropagationParams(
        m_geff_ev=m_geff_ev,
        distance_gpc=luminosity_distance / 1e3,
        foam_strength=foam_strength,
        rng_seed=kwargs.get("rng_seed"),
        foam_corr_m=foam_corr_m if foam_corr_m is not None else kwargs.get("foam_corr_m"),
    )
    hp, hc = hybrid_waveform(
        mass_1,
        mass_2,
        delta_t=delta_t,
        f_lower=f_lower,
        params=params,
        distance_mpc=luminosity_distance,
        inclination=theta_jn,
        coa_phase=phase,
        approximant=approximant,
        return_cross=True,
    )
    hp = _ensure_sampling(hp, delta_t)
    hc = _ensure_sampling(hc, delta_t)
    target_len = len(time_array)
    plus = _match_length(hp, target_len)
    cross = _match_length(hc, target_len)
    return {"plus": plus, "cross": cross}


def inject_into_strain(
    strain: TimeSeries,
    waveform: TimeSeries,
    t_ref: float,
    event_start: Optional[float] = None,
) -> TimeSeries:
    """Inject ``waveform`` into ``strain`` aligned to ``t_ref`` seconds."""
    strain_copy = strain.copy()
    if event_start is not None:
        strain_copy.start_time = event_start

    start_time = float(strain_copy.start_time)
    delta_t = float(strain_copy.delta_t)
    waveform_start = float(waveform.start_time)
    injection_start_time = t_ref + waveform_start

    start_index = int(round((injection_start_time - start_time) / delta_t))
    end_index = start_index + len(waveform)

    if start_index < 0 or end_index > len(strain_copy):
        raise ValueError("Injection window falls outside strain data")

    strain_copy.data[start_index:end_index] += waveform.data
    return strain_copy


def whiten_strain(strain: TimeSeries, fftlength: float = 4.0) -> TimeSeries:
    """Whiten strain using gwpy's implementation for quick prototyping."""
    gwpy_series = GwpyTimeSeries(strain.numpy(), dt=strain.delta_t, t0=strain.start_time)
    whitened = gwpy_series.whiten(fftlength=fftlength)
    return TimeSeries(whitened.value, delta_t=whitened.dt.value, epoch=whitened.t0.value)


def gwpy_to_pycbc(series: GwpyTimeSeries) -> TimeSeries:
    """Convert a `gwpy` TimeSeries to PyCBC TimeSeries."""
    return TimeSeries(series.value, delta_t=series.dt.value, epoch=series.t0.value)


def pycbc_to_gwpy(series: TimeSeries) -> GwpyTimeSeries:
    """Convert a PyCBC TimeSeries to `gwpy` TimeSeries."""
    return GwpyTimeSeries(series.numpy(), dt=series.delta_t, t0=series.start_time)


if __name__ == "__main__":
    params = HybridPropagationParams(m_geff_ev=1e-23, distance_gpc=1.0, foam_strength=1.0, rng_seed=1234)
    waveform = hybrid_waveform(30, 30, params=params)
    print(waveform.sample_times[:5])
    print(waveform.data[:5])
