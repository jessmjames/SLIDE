# SLIDE

## Sequence-free Landscape Inference for Directed Evolution

Directed evolution is a method for engineering biological systems or components, such as proteins, wherein desired traits are optimised through iterative rounds of mutagenesis and selection of fit variants. The process of protein directed evolution can be envisaged as navigation over high-dimensional landscapes with numerous local maxima. The performance of any strategy in navigating such a landscape is dependent on the ruggedness of that landscape. However, this information is generally unavailable at the outset of an experiment, and cannot currently be computed using analytical methods. 

Here we propose **SLIDE**, **S**equence-free **L**andscape **I**nference for **D**irected **E**volution, a method for estimating landscape ruggedness from a mutating population, using only population-level phenotypic data and knowledge of mutation rate. This method uses a short period of exploration at the beginning of an experiment to predict the ruggedness, subsequently guiding the choice of high-performing parameters for directed evolution control.


## 📂 Repository Structure

SLIDE/\
├── scripts/ # Scripts to produce the files (which can be downloaded from 10.5281/zenodo.16849761)\
├── plot_data/ # Minimal data required for producing plots in ruggedness_figures_plots.ipynb\
├── landscape_arrays/ # Scripts for running analyses\
├── direvo_functions.py # Directed evolution functions\
├── selection_function_library.py # Selection functions\
├── ruggedness_functions.py # Functions for ruggedness analysis\
├── ruggedness_figures_data_processing.ipynb # Pre-processing to produce data in plot_data/\
├── ruggedness_figures_plots.py # Code for producing plots\
└── README.md # This file\