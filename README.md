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

## Instructions

The refactored pipeline is notebook-driven. Run the notebooks from the repository root in this order:

1. `data_generation.ipynb`
   - Generates raw simulation products.
   - Creates `raw_data/` if needed.

2. `data_processing.ipynb`
   - Processes raw products from `raw_data/` into plotting inputs.
   - Creates `processed_data/` if needed.

3. `data_visualisation.ipynb`
   - Loads plotting inputs from `processed_data/`.
   - Saves figures into `figures/pdf/`, `figures/eps/`, and `figures/png/`.


## Repository Structure

```text
SLIDE/
├── data_generation.ipynb          # Generate raw simulation data into raw_data/
├── data_processing.ipynb          # Process raw data into processed_data/
├── data_visualisation.ipynb       # Create figures in figures/{pdf,eps,png}/
├── environment.yml                # Conda/mamba environment definition for SLIDE_env
├── environment_cpu.yml            # CPU-only Conda/mamba environment definition for SLIDE_env_cpu
├── slide/                         # Refactored reusable library code
│   ├── data_generation.py         # Raw-data simulation and product registry
│   ├── data_processing.py         # Processing helpers for plotting inputs
│   ├── direvo_functions.py        # Directed-evolution and diffusion routines
│   ├── ruggedness_functions.py    # Ruggedness and spectral analysis helpers
│   ├── selection_function_library.py
│   └── utils.py                   # Paths, pickle I/O, filename and figure helpers
├── landscape_arrays/              # Empirical landscape arrays
├── other_data/                    # Small auxiliary data files
├── raw_data/                      # Generated raw products; created by data_generation.ipynb
├── processed_data/                # Generated plotting inputs; created by data_processing.ipynb
├── figures/                       # Generated figures; created by data_visualisation.ipynb
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

[Steel Lab Oxford](http://steel.ac/) | [Control Group](https://eng.ox.ac.uk/control)
