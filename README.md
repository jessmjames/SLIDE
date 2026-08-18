# SLIDE

## Sequence-free Landscape Inference for Directed Evolution

Directed evolution is a method for engineering biological systems or components, such as proteins, wherein desired traits are optimised through iterative rounds of mutagenesis and selection of fit variants. The process of protein directed evolution can be envisaged as navigation over high-dimensional landscapes with numerous local maxima. The performance of any strategy in navigating such a landscape is dependent on the ruggedness of that landscape. However, this information is generally unavailable at the outset of an experiment, and cannot currently be computed using analytical methods. 

Here we propose **SLIDE**, **S**equence-free **L**andscape **I**nference for **D**irected **E**volution, a method for estimating landscape ruggedness from a mutating population, using only population-level phenotypic data and knowledge of mutation rate. This method uses a short period of exploration at the beginning of an experiment to predict the ruggedness, subsequently guiding the choice of high-performing parameters for directed evolution control.

## Installation

Create the CUDA-enabled Conda environment with mamba:

```bash
mamba env create -f environment.yml
conda activate SLIDE_env
```

Alternatively, use conda:

```bash
conda env create -f environment.yml
conda activate SLIDE_env
```

For a CPU-only installation, use `environment_cpu.yml` instead and activate `SLIDE_env_cpu`.  

To make the environment available as a Jupyter notebook kernel, run:

```bash
python -m ipykernel install --user --name SLIDE_env --display-name "Python (SLIDE_env)"
```

For the CPU-only environment, run:

```bash
python -m ipykernel install --user --name SLIDE_env_cpu --display-name "Python (SLIDE_env_cpu)"
```

JAX is included in both environment files. `environment.yml` installs a CUDA-enabled JAX build, while `environment_cpu.yml` installs a CPU-only JAX build.

## Data Download

Download the cached raw simulation products and processed plotting inputs from
the repository root with:

```bash
python slide/download_zenodo_data.py
```

The script downloads `raw_data.zip` and `processed_data.zip` from the configured
public data URL and extracts them into `raw_data/` and `processed_data/`. If an
extracted file already exists, the script asks before overwriting it. Pass
`--force` to overwrite existing files without prompting:

```bash
python slide/download_zenodo_data.py --force
```

These cached products allow the figure notebooks to process or plot the paper
figures without repeating the longest simulations. The notebooks can still
generate missing products when their overwrite and plot-only settings permit it.

## Instructions

The analysis is figure-notebook driven. Run notebooks from the repository root so
their relative script paths and data-directory helpers resolve correctly.

For an introductory example of generating mutation-only fitness decay and fitting
local and global ruggedness estimates, run `slide_workflow.ipynb`.

Primary figure notebooks:

- `figure_3.ipynb`
- `figure_4.ipynb`
- `figure_5.ipynb`
- `figure_6.ipynb`
- `figure_7.ipynb`

Supplemental figure notebooks:

- `figure_S2.ipynb`
- `figure_S3.ipynb`
- `figure_S4.ipynb`

Each figure notebook owns its raw-data checks, processing, and plotting. Set the
notebook flags such as `PLOT_ONLY`, `OVERWRITE_RAW_PKL`, and
`OVERWRITE_PROCESSED_PKL` as needed before running. Raw products are stored in
`raw_data/`, processed plotting inputs in `processed_data/`, and saved figures in
`figures/{pdf,eps,png}/`.


## Repository Structure

```text
SLIDE/
├── slide_workflow.ipynb           # Introductory SLIDE analysis workflow
├── figure_3.ipynb                 # Primary figure notebook
├── figure_4.ipynb                 # Primary figure notebook
├── figure_5.ipynb                 # Primary Figure 5 notebook
├── figure_6.ipynb                 # Primary Figure 6 notebook
├── figure_7.ipynb                 # Primary Figure 7 notebook
├── figure_S2.ipynb                # Supplemental figure notebook
├── figure_S3.ipynb                # Supplemental figure notebook
├── figure_S4.ipynb                # Supplemental figure notebook
├── environment.yml                # Conda/mamba environment definition for SLIDE_env
├── environment_cpu.yml            # CPU-only Conda/mamba environment definition for SLIDE_env_cpu
├── scripts/                       # Long-running raw-data generators used by figures
├── slide/                         # Reusable library code
│   ├── data_generation.py         # Raw-data simulation helpers
│   ├── direvo_functions.py        # Directed-evolution and diffusion routines
│   ├── ruggedness_functions.py    # Ruggedness and spectral analysis helpers
│   ├── selection_function_library.py
│   └── utils.py                   # Paths, pickle I/O, filename and figure helpers
├── landscape_arrays/              # Empirical landscape arrays
├── other_data/                    # Small auxiliary data files
├── raw_data/                      # Generated raw products
├── processed_data/                # Generated plotting inputs
├── figures/                       # Generated figures
└── README.md
```


## Citation

If you use SLIDE, please cite:

Towers S, James J, Steel H, Kempf I (2026) Sequence-free landscape inference for directed evolution. *PLOS Computational Biology*. Volume, issue, article number, and DOI to be added.

```bibtex
@article{towers2026slide,
  title = {Sequence-free landscape inference for directed evolution},
  author = {Towers, Sebastian and James, Jessica and Steel, Harrison and Kempf, Idris},
  journal = {PLOS Computational Biology},
  year = {2026},
  volume = {TBD},
  number = {TBD},
  pages = {TBD},
  doi = {TBD},
  url = {TBD}
}
```

[Steel Lab Oxford](http://steel.ac/) | [Applied Control Laboratory](https://users.ox.ac.uk/~lady5906/) | [Control Group](https://eng.ox.ac.uk/control)
