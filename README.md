# fluorokinetics

Code for **"Cell-to-cell heterogeneity and identifiability of apparent fluorescence-decay kinetics in carbon-starved *Escherichia coli*"** (S. Das).

The pipeline reanalyses the single-cell degradation-calibration data of Gervais *et al.* (2026, *Mol. Syst. Biol.* 22:1531–1550, doi:10.1038/s44320-026-00226-5). It estimates cell-to-cell heterogeneity in apparent fluorescence-decay kinetics with a hierarchical model, and reproduces every number, table and figure in the paper.

## What it does

- **Hierarchical model.** log F_i(t) = a_i − (γ_i/ω)(1 − e^(−ωt)) + ε_i(t), where γ_i is a cell-specific initial apparent decay rate (lognormal), ω is a group-level relaxation rate, and ε_i is stationary Ornstein–Uhlenbeck (OU) noise. Each cell's intercept is profiled out by generalised least squares.
- **Adaptive Gauss–Hermite quadrature** for the random effects. It is validated against brute-force integration. Fixed-node quadrature is avoided because it produces spurious likelihood optima for these sharply peaked single-cell likelihoods.
- **Analyses:**
  - cellwise model comparison and profile likelihood
  - synthetic identifiability benchmark with OU noise
  - trajectory quality control (QC) at several thresholds
  - likelihood-ratio tests of a time-varying vs constant rate, and of random vs common γ₀
  - lane random effects and a gamma random effect
  - parametric and lane-cluster bootstraps
  - photobleaching correction
  - cell-level covariates
  - simulation-based validation of the whole pipeline (parameter recovery), including artefacts, lane effects, model misspecification and the published ω values

## Installation

Requires Python ≥ 3.9.
```bash
git clone https://github.com/Soutrick93/fluorokinetics.git
cd fluorokinetics
python -m pip install -r requirements.txt
```

## Data

Download the data archive of Gervais *et al.* (2026) and copy these two files into the repository folder:

| File | Used by |
|---|---|
| `myframes_degradation.feather` | all stages |
| `myframes_bleaching.feather` | `bleaching` stage only |

Other locations can be set in `config.json`, or with `--data PATH` and `--bleaching PATH`. A `.csv` or `.csv.gz` export of the degradation table also works. Trajectories are selected automatically: the starvation phase `SPM.1`, with at least 8 positive time points. This gives the 393 cells analysed in the paper.

## Usage

```bash
python run_all.py --quick --jobs 4      # about 5 min check on two small groups, writes results_quick/
python run_all.py --jobs 4              # full analysis, writes results/ and figures/
python run_all.py --list                # stages, inputs and outputs
python run_all.py --stages main,figures # selected stages only
```

Approximate times on 4 cores:

| Stages | Time |
|---|---|
| `cellwise, main, loo, synthetic, covariates, bleaching` | about 10 min |
| `structure` (lane and gamma models) | about 30 to 60 min |
| `bootstrap` (1000 + 500 parametric, 500 lane-cluster replicates) | several hours |
| `sbc` (simulation-based validation) | about 1 to 2 h |

Interrupted runs resume automatically: finished stages are skipped, and the bootstrap and SBC stages continue from their checkpoints. `--force` re-runs a stage from scratch and deletes its previous output. `--pack` writes all results to `results_archive.zip`.

## Stages and outputs

| Stage | Outputs (`results/<stage>/`) | Paper |
|---|---|---|
| `cellwise` | `cellwise.csv`, `cellwise_summary.csv` | Table 1, Fig. 2, Fig. 3A |
| `main` | `hierarchical_fits.csv`, `model_comparison.csv`, `theta.json` | Table 3, Tables S1 and S2 |
| `loo` | `leave_one_out.csv` | influence analysis |
| `synthetic` | `synthetic_summary.csv` | Table 2, Fig. 3A |
| `structure` | `lane_fits.csv`, `gamma_fits.csv`, `structure_comparison.csv` | Table 4, Fig. 5 |
| `bootstrap` | `parametric_bootstrap_summary.csv`, `lane_bootstrap_summary.csv` | confidence intervals |
| `bleaching` | `bleaching_calibration.csv`, `bleaching_refits.csv` | Table S3 |
| `covariates` | `cell_estimates.csv`, `covariate_summary.csv` | Table S4, Figs 1 and 4 |
| `sbc` | `sbc_replicates.csv`, `sbc_summary.csv` | Table 5, Fig. S3 |
| `figures` | `figures/*.svg`, `.pdf`, `.png` | all figures |

## Repository structure

```
run_all.py          driver (stages, resume, parallel jobs)
config.json         paths and all analysis settings
fk/data.py          loading, trajectory selection, quality control
fk/likelihood.py    OU likelihood, adaptive quadrature, lane and gamma models
fk/fit.py           multi-start maximum likelihood
fk/cellwise.py      cellwise fits, profile likelihood, synthetic benchmark
fk/stages.py        analysis stages
fk/figures.py       publication figures
fk/util.py          parallel map, checkpoints, logging
tests/              numerical checks (pytest)
```

## Tests

```bash
python -m pytest tests -q
```
The tests check the adaptive quadrature against brute-force integration (lognormal, gamma and lane models) and verify that a simulated dataset is recovered. They need no data files.

## Citation

If you use this code, please cite the paper and the original data:

- Das S. Cell-to-cell heterogeneity and identifiability of apparent fluorescence-decay kinetics in carbon-starved *Escherichia coli*. (submitted)
- Gervais T, Kscheschinski B, Mell M, Goepfert N, van Nimwegen E, Julou T. *E. coli* prepares for starvation by dramatically remodeling its proteome in the first hours after loss of nutrients. *Mol. Syst. Biol.* 22, 1531–1550 (2026).

## License

MIT (see `LICENSE`).
