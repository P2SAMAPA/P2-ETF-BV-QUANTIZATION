# Batalin–Vilkovisky (BV) Quantization Engine for ETFs

Applies quantum gauge field theory to ETF correlation matrices using the BV formalism. The **ghost number anomaly** per ETF measures structural essentiality — how much an ETF cannot be “gauged away”.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63, 252, 504, 1008, 2016, 4032, 4536 days)
- Spectral decomposition of correlation matrix → positive/zero/negative eigenvalue modes
- Ghost number per mode: +1 (positive), 0 (zero), –1 (negative)
- Per‑ETF ghost number = sum over modes (ghost_number × eigenvector component²)
- Best window automatically selected (largest absolute raw signal)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-bv-quantization-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- **Positive ghost number** → ETF projects onto positive‑eigenvalue modes → structurally essential, cannot be eliminated by redefining variables.
- **Negative ghost number** → ETF projects onto negative‑eigenvalue modes → gauge‑redundant, may be removed.
- **Zero ghost number** → ETF lies in the physical subspace after gauge fixing.
- This is the first application of BV quantization to quantitative finance.

## Requirements

See `requirements.txt`.
