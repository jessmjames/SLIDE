# Rationalisation Checklist

Goal: every figure gets a folder in `figures_src/` with three files:
- `1_raw_data.py` — runs simulations → `SLIDE_data/`
- `2_process_data.py` — `SLIDE_data/` → `plot_data/*.pkl`
- `3_plot.py` — `plot_data/*.pkl` → `figures/*.pdf`

See `docs/figure_pipelines.md` for full pipeline details per figure.

---

## Quick fixes (do first)

- [x] **Fig 6** — change `savefig` output in `scripts/plot_mutation_model_comparison.py` from `accuracy_over_sampling_nuc_models.pdf` to `graph_mutation_model.pdf`
- [x] **S4** — plotting extracted into `figures_src/figure_S4_strategy_gens/3_plot.py` (scripts, not notebook)
- [x] **4C** — violin plot extracted into `figures_src/figure_4C_repeat_heterogeneity/3_plot.py` (scripts, not notebook)

---

## Phase 1: figures with existing standalone scripts

These are closest to the target structure — just need organising into folders.

- [x] **S2** `rho_accuracy_mutation_models.pdf` — `figures_src/figure_S2_rho_accuracy/`
- [x] **S3** `decay_curves_nuc_75steps.pdf` — `figures_src/figure_S3_decay_curves_nuc/`
- [x] **Fig 6** `graph_mutation_model.pdf` — `figures_src/figure_6_mutation_model/`
- [x] **Fig 4C** `repeat_heterogeneity.pdf` — `figures_src/figure_4C_repeat_heterogeneity/`
- [x] **S4** `strategy_prediction_{25,50,100}gens.pdf` — `figures_src/figure_S4_strategy_gens/`

---

## Phase 2: notebook-only figures (extract into scripts)

For each figure: identify the relevant notebook cells, trace data dependencies,
write the three scripts.

- [ ] **Fig 2A–2E** (3D landscape visualisations) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 3A–3E** (NK accuracy plots) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 4A–4B, 4D–4F** (empirical accuracy/variance) — `ruggedness_figures_plots.ipynb`
- [x] **Fig 4C** (repeat heterogeneity — violin) — `figures_src/figure_4C_repeat_heterogeneity/`
- [ ] **Fig 4G** (empirical ruggedness comparison IK) — `ruggedness_figures_plots_IK.ipynb`
- [ ] **Fig 5A–5G** (strategy prediction + DE) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 7** (NK spectra) — `ruggedness_figures_plots.ipynb`
- [x] **S4** (strategy prediction over generations) — `figures_src/figure_S4_strategy_gens/`

---

## Phase 3: figures with scattered/incomplete code

- [ ] **S1** `alt_landscapes.pdf`
  - [ ] Merge `decay_curve_sweep_alt_landscapes_*.py` and `strategy_sweep_alt_landscapes_*.py` from `rebuttals` into main
  - [ ] Extract plotting cell from `ruggedness_figures_data_processing.ipynb` (`rebuttals`)
  - [ ] Create full 3-file structure in `figures_src/figure_S1_alt_landscapes/`
