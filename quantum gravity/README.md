# Hybrid Quantum Foam Graviton Mass Pipeline

This repository contains the code used in *Hybrid Quantum Foam-Induced Graviton Mass Signatures in Gravitational-Wave Propagation*, providing a reproducible path from effective field theory derivations to Bayesian parameter estimation on public LIGO/Virgo data.

## Repository Layout

- `analysis_scaffold.py` – end-to-end matched-filter and Bilby inference pipeline.
- `waveform_hybrid.py` – waveform utilities implementing foam-induced phase/amplitude corrections.
- `hybrid_mass_model.py` – symbolic derivations for loop-induced graviton masses.
- `foam_coherence_calibration.py` – coherence length sweep diagnostics.
- `summarize_bilby.py` – posterior summary helper for Bilby results.
- `paper.tex` – manuscript (to be updated to REVTeX in workflow below).
- `bilby_stub_*` – example Bilby output directories for GW events.

## Public Data

All gravitational-wave strain segments are fetched directly from the Gravitational-Wave Open Science Center (GWOSC). The default configuration downloads 32-second stretches (sample rate 4096 Hz) around GW190521 for the Hanford (H1) and Livingston (L1) interferometers (~1.6 MiB per detector segment). No proprietary data are stored in this repository.

## Quick Start

```bash
conda env create -f environment.yml
conda activate quantum-foam-gw
python3 analysis_scaffold.py
```

The script attempts to analyse GW230529_181500; if GWTC-4 strain is not yet public it falls back to GW190521, injecting the hybrid waveform and running Bilby.

## Reproducing Key Results

1. **Baseline run:** `python3 analysis_scaffold.py` – generates matched-filter SNRs and Bilby samples in `bilby_stub_GW190521/`.
2. **Posterior summary:** `python3 summarize_bilby.py bilby_stub_GW190521/hybrid_result.json`.
3. **Coherence sweep:** `python3 foam_coherence_calibration.py`.

## Citation

If you use this code, please cite the accompanying paper and the GWOSC data release.

