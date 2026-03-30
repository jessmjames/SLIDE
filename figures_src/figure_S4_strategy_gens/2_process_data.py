"""
Figure S4 — Step 2: compute optimal strategies per generation.

Loads per-generation strategy sweep data and extracts optimal base_chance
and population_split for each K value at each generation count.

Input
-----
  SLIDE_data/N4A20_strategy_sweep_{g}_gens.pkl  for g in [5, 25, 50, 75, 100]

Output
------
  plot_data/strategy_gens_data.pkl
    all_strategy_data : (5, 3, 7, 7) — [gen, K, split, base_chance]
    gens              : [5, 25, 50, 75, 100]
    splits            : [24, 20, 16, 12, 8, 4, 1]
    base_chances      : array of 7 values from base_chance_threshold_fixed_prop

Usage
-----
  python figures_src/figure_S4_strategy_gens/2_process_data.py
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
from direvo_functions import base_chance_threshold_fixed_prop
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

GENS = [5, 25, 50, 75, 100]

strategy_data_by_gen = []
for g in GENS:
    path = os.path.join(slide_data_dir, f'N4A20_strategy_sweep_{g}_gens.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    strategy_data_by_gen.append(data.mean(axis=0))  # average over reps

all_strategy_data = np.array(strategy_data_by_gen)  # (5, 3, 7, 7)

_, base_chances = base_chance_threshold_fixed_prop([0, 0.19], 0.2, 7)
splits          = [24, 20, 16, 12, 8, 4, 1]

out = os.path.join(plot_data_dir, 'strategy_gens_data.pkl')
with open(out, 'wb') as f:
    pickle.dump({
        'all_strategy_data': all_strategy_data,
        'gens':              GENS,
        'splits':            splits,
        'base_chances':      base_chances,
    }, f)
print(f'Saved → {out}')
