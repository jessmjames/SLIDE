# Instructions
- Use the conda/mamba environments called SLIDE_env or SLIDE_env_cpu.
- Do not add or commit or delete anything from git.
- Always add doc strings with parameters/returns, and add type hinst avoiding use of Any.
- Avoid wrappers and helpers where ever possible.

Instructions specific to figure notebook creation/modification:
- Use figure_4.ipynb as an example:
  - Seperate raw data generation and processed data generation.
  - If not produced in other notebooks, generate all required data in the notebook.
  - Figure notebooks must be self-contained: when raw or processed products are unavailable, and the overwrite/plot-only flags allow it, the notebook should generate or process those products itself.
  - Parallel or per-landscape notebooks may be created as temporary data-generation helpers, but the main figure notebook must still be able to generate and process the required data.
  - Structure the notebook in similar ways with individual figure panels and final figure in the end of the notebook.
  - Add and implement the flags OVERWRITE_RAW_PKL, OVERWRITE_PROCESSED_PKL, and PLOT_ONLY to each figure notebook.
