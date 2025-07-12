# Paper Experiments Guide

This guide explains how to run experiments and generate figures for the paper comparing different POMDP solvers on project end-time estimation problems.

## Overview

The experiment pipeline consists of two main stages:
1. **Data Collection**: Run experiments and save raw results
2. **Analysis**: Generate plots and tables from saved results

## Quick Start

First Generate Experiments. The data in `reference_problems/std_div_3` was generated with the following commands .

Note that for the `-l 2 -u 52` "large" problems they were generated over multiple runs to fit into the memory of the machines (a 512 GB server). You could do it on smaller machines, but you might need to further split the runs into smaller chunks.

```bash
## Generate Data

# Small
julia --project bin/cli.jl evaluate -r 42 --solvers QMDP -n 1000 -l 2 -u 12 --no-plot

# Medium
julia --project bin/cli.jl evaluate -r 42 --solvers QMDP -n 1000 -l 2 -u 26 --no-plot

# Large
julia --project bin/cli.jl evaluate -r 42 --solvers QMDP -n 250 -l 2 -u 52 --no-plot
julia --project bin/cli.jl evaluate -r 43 --solvers QMDP -n 250 -l 2 -u 52 --no-plot
julia --project bin/cli.jl evaluate -r 44 --solvers QMDP -n 250 -l 2 -u 52 --no-plot
julia --project bin/cli.jl evaluate -r 45 --solvers QMDP -n 250 -l 2 -u 52 --no-plot

## Merge Data to create smaller output files to commit

julia --project scripts/merge_simulation_data.jl ./results/std_dev_3_qmdp_l_2_u_12_n_1000_s42/evaluation_results.json -o qmdp_base_l_2_u_12_n_1000.json
julia --project scripts/merge_simulation_data.jl ./results/std_dev_3_qmdp_l_2_u_26_n_1000_s42/evaluation_results.json -o qmdp_base_l_2_u_26_n_1000.json
julia --project scripts/merge_simulation_data.jl -o qmdp_base_l_2_u_52_n_1000 ./results/std_dev_3_qmdp_l_2_u_52_n_250_s42/evaluation_results.json ./results/std_dev_3_qmdp_l_2_u_52_n_250_s43/evaluation_results.json ./results/std_dev_3_qmdp_l_2_u_52_n_250_s44/evaluation_results.json ./results/std_dev_3_qmdp_l_2_u_52_n_250_s45/evaluation_results.json
```

```bash
# Run all experiments (this will take some time)
julia --project scripts/run_paper_experiments.jl

# Analyze results (Passing in the experiment directory displayed above))
julia --project scripts/analyze_paper_results.jl PATH/TO/EXPERIMENT/DIRECTORY/

```

## Detailed Usage

### 1. Running Experiments

The main experiment script compares 5 solvers across 3 problem sizes:

**Solvers:**
- `OBSERVEDTIME`: Simple baseline that announces the observed time
- `MOSTLIKELY`: Announces based on most likely belief state
- `QMDP`: QMDP approximation algorithm
- `CXX_SARSOP`: Point-based belief-space POMDP solver (C++ implementation)
- `MOMDP_SARSOP`: SARSOP solver using MOMDP formulation

**Problem Sizes:**
- Small: 2-12 timesteps
- Medium: 2-26 timesteps  
- Large: 2-52 timesteps

To run experiments with custom settings:

```julia
using POMDPPlanning

# Load problem configurations
problem_configs = POMDPPlanning.load_problem_configs("reference_problems")

# Run experiments
experiment_dir, results = POMDPPlanning.run_paper_experiments(
    problem_configs,
    ["OBSERVEDTIME", "MOSTLIKELY", "QMDP", "CXX_SARSOP", "MOMDP_SARSOP"],
    "output_directory",
    num_simulations = 100,      # Simulations per solver/problem
    num_detailed_plots = 15,    # Number of belief evolution plots to save
    policy_timeout = 300,       # Timeout for SARSOP solvers (seconds)
    seed = 42,                  # Random seed for reproducibility
    verbose = true
)
```

### 2. Analyzing Results

To analyze previously saved results:

```bash
julia scripts/analyze_paper_results.jl path/to/experiment_directory
```

Or programmatically:

```julia
include("scripts/analyze_paper_results.jl")
analyze_results("path/to/experiment_directory")
```

## Output Structure

After running experiments and analysis, you'll find:

```
paper_results/
└── paper_experiment_YYYY-MM-DD_HH-MM-SS/
    ├── experiment_config.json          # Experiment configuration
    ├── results_small.json              # Raw results for small problems
    ├── results_medium.json             # Raw results for medium problems
    ├── results_large.json              # Raw results for large problems
    ├── all_results.json                # Combined results
    ├── belief_evolution_plots/         # Detailed belief evolution plots
    │   ├── small/
    │   ├── medium/
    │   └── large/
    └── analysis/                       # Generated plots and tables
        ├── reward_statistics.csv       # Reward statistics table
        ├── reward_table.tex            # LaTeX formatted reward table
        ├── comparison_statistics.csv   # Performance metrics table
        ├── statistics_table.tex        # LaTeX formatted statistics table
        ├── reward_comparison_*.png     # Reward comparison plots
        ├── histograms/                 # Reward distribution histograms
        ├── statistics_plots/           # Performance metric plots
        └── combined_plots/             # Multi-metric comparison plots
```

## Key Outputs for Paper

### Tables
1. **reward_table.tex**: Mean rewards with standard deviations by solver and problem size
2. **statistics_table.tex**: Comprehensive performance metrics including:
   - Average number of announcement changes
   - Standard deviation of announcement changes
   - Average final error
   - Percentage of incorrect final predictions
   - Average magnitude of announcement changes
   - Policy generation time

### Figures
1. **Reward Comparisons**:
   - `reward_comparison_small.png`, `_medium.png`, `_large.png`: Bar charts with error bars
   - `reward_comparison_combined.png`: Grouped bar chart across all problem sizes

2. **Reward Distributions**:
   - `histograms/hist_*_*.png`: Individual histograms for each solver/problem combination
   - `histograms/hist_*_combined.png`: Overlaid histograms by problem size
   - `histograms/hist_*_all_sizes.png`: Overlaid histograms by solver

3. **Performance Metrics**:
   - `statistics_plots/avg_announcement_changes.png`: Changes over problem sizes
   - `statistics_plots/avg_final_error.png`: Final error over problem sizes
   - `statistics_plots/incorrect_predictions.png`: Error rates over problem sizes

4. **Combined Analysis**:
   - `combined_plots/key_metrics_comparison.png`: 2x2 subplot of key metrics

5. **Belief Evolution**:
   - `belief_evolution_plots/*/`: 2D heatmaps showing belief evolution with announced times overlaid

## Customization

### Changing Timeout

Edit `POLICY_TIMEOUT` in `scripts/run_paper_experiments.jl`:
```julia
POLICY_TIMEOUT = 600  # 10 minutes instead of 5
```

### Changing Number of Simulations

Edit `NUM_SIMULATIONS` in `scripts/run_paper_experiments.jl`:
```julia
NUM_SIMULATIONS = 200  # More simulations for smoother statistics
```

### Adding New Metrics

Modify `generate_statistics_table()` in `scripts/analyze_paper_results.jl` to compute additional metrics from the raw simulation data.

## Reproducing Results

All experiments use fixed random seeds from previous experiment runs for reproducibility.