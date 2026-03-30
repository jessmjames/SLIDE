"""
Figure S4 — Step 1: strategy sweep over generations for N=4, A=20 landscapes.

Runs directed evolution strategy sweeps at different generation counts,
sweeping over base_chance and population_split parameters.

Output
------
  SLIDE_data/N4A20_strategy_sweep_{g}_gens.pkl  for g in [5, 25, 50, 75, 100]

Usage
-----
  python figures_src/figure_S4_strategy_gens/1_raw_data.py

Note: requires a GPU. The sweep scripts for these files are the untracked
scripts/strategy_sweep_N10_A20_gens.py variants — confirm which script
generated the N4A20 data before running.

Shared raw data: none (unique to S4).
"""

# TODO: identify and commit the correct generation sweep script,
# then call it here via subprocess.
print("Raw data: SLIDE_data/N4A20_strategy_sweep_{g}_gens.pkl for g in [5,25,50,75,100].")
print("Sweep script not yet confirmed — check scripts/strategy_sweep_N10_A20_gens.py variants.")
