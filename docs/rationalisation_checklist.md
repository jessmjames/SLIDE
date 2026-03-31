# Rationalisation Checklist

Goal: every figure gets a folder in `figures_src/` with three files:
- `1_raw_data.py` — runs simulations → `SLIDE_data/`
- `2_process_data.py` — `SLIDE_data/` → `plot_data/*.pkl`
- `3_plot.py` — `plot_data/*.pkl` → `figures/*.pdf`

See `docs/figure_pipelines.md` for full pipeline details per figure.

---

## Quick fixes (do first)

- [x] **Fig 6** — change `savefig` output in `scripts/plot_mutation_model_comparison.py` from `accuracy_over_sampling_nuc_models.pdf` to `graph_mutation_model.pdf`
- ⚠️ **S4** — source unclear; see note in `docs/figure_pipelines.md`. Awaiting confirmation from Jess.
- [x] **4C** — violin plot extracted into `figures_src/figure_4C_repeat_heterogeneity/3_plot.py` (scripts, not notebook)

---

## Phase 1: figures with existing standalone scripts

These are closest to the target structure — just need organising into folders.

- [x] **S2** `rho_accuracy_mutation_models.pdf` — `figures_src/figure_S2_rho_accuracy/`
- [x] **S3** `decay_curves_nuc_75steps.pdf` — `figures_src/figure_S3_decay_curves_nuc/`
- [x] **Fig 6** `graph_mutation_model.pdf` — `figures_src/figure_6_mutation_model/`
- [x] **Fig 4C** `repeat_heterogeneity.pdf` — `figures_src/figure_4C_repeat_heterogeneity/`
- ⚠️ **S4** `strategy_prediction_{g}gens.pdf` — source disputed; see `docs/figure_pipelines.md`. Current `figures_src/figure_S4_strategy_gens/3_plot.py` outputs heatmaps which may be wrong.

---

## Phase 2: notebook-only figures (extract into scripts)

For each figure: identify the relevant notebook cells, trace data dependencies,
write the three scripts.

- [ ] **Fig 2A–2E** (3D landscape visualisations) — `ruggedness_figures_plots.ipynb`
- [x] **Fig 3A** `decay_curves_example.pdf` — `figures_src/figure_3A_decay_curves_example/` ✓ verified
- [x] **Fig 3B** `accuracy_over_K.pdf` — `figures_src/figure_3B_accuracy_over_K/` ✓ verified
- [x] **Fig 3C** `accuracy_over_popsize.pdf` — `figures_src/figure_3C_accuracy_over_popsize/` ✓ verified
- [x] **Fig 3D** `accuracy_over_mut.pdf` — `figures_src/figure_3D_accuracy_over_mut/` ✓ verified
- [x] **Fig 3E** `NK_ruggedness_metric_comparison.pdf` — `figures_src/figure_3E_NK_ruggedness_comparison/` ✓ verified
- [ ] **Fig 4A–4B, 4D–4F** (empirical accuracy/variance) — `ruggedness_figures_plots.ipynb`
- [x] **Fig 4C** (repeat heterogeneity — violin) — `figures_src/figure_4C_repeat_heterogeneity/`
- [ ] **Fig 4G** (empirical ruggedness comparison IK) — `ruggedness_figures_plots_IK.ipynb`
- [x] **Fig 5A** `strategy_prediction.pdf` — `figures_src/figure_5A_strategy_prediction/` ✓ verified
- [ ] **Fig 5B–5G** (NK DE + empirical DE) — `ruggedness_figures_plots.ipynb`
- ⚠️ **Fig 7** `NK_spectra.pdf` — `figures_src/figure_7_NK_spectra/` extracted but **cannot run**: `np.concat` bug in `get_single_decay_rate` (cell 80 of `ruggedness_figures_plots.ipynb` on main). Fix needed in notebook before script will work. Flag to Jess.
- ⚠️ **S4** (source unclear — awaiting Jess) — `figures_src/figure_S4_strategy_gens/` exists but plotting may be wrong

---

## Phase 3: figures with scattered/incomplete code

- [ ] **S1** `alt_landscapes.pdf`
  - [ ] Merge `decay_curve_sweep_alt_landscapes_*.py` and `strategy_sweep_alt_landscapes_*.py` from `rebuttals` into main
  - [ ] Extract plotting cell from `ruggedness_figures_data_processing.ipynb` (`rebuttals`)
  - [ ] Create full 3-file structure in `figures_src/figure_S1_alt_landscapes/`
