# SLIDE — Claude Context

## Project docs

Detailed documentation lives in `docs/`. Read the relevant file when you need specifics.

| File | Contents |
|------|----------|
| [docs/figure_pipelines.md](docs/figure_pipelines.md) | Step-by-step pipelines to regenerate all figures from raw simulations |
| [docs/rationalisation_checklist.md](docs/rationalisation_checklist.md) | Checklist for rationalising codebase into per-figure 3-file structure |

## Repo structure (brief)

- `scripts/` — standalone Python scripts for simulations and plotting
- `landscape_arrays/` — static empirical fitness landscape arrays
- `other_data/` — mutation matrices and other static inputs
- `plot_data/` — preprocessed intermediate data (generated, not tracked by git)
- `figures/` — output figures (generated, not tracked by git)
- `SLIDE_data/` — large simulation outputs; external directory via `slide_config.get_slide_data_dir()`
