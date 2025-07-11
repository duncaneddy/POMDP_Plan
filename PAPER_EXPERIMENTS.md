# Paper Experiments Guide

This guide explains how to run experiments and generate figures for the paper comparing different POMDP solvers on project end-time estimation problems.

## Overview

The experiment pipeline consists of two main stages:
1. **Data Collection**: Run experiments and save raw results
2. **Analysis**: Generate plots and tables from saved results

## Quick Start

```bash
# Run all experiments (this will take some time)
julia scripts/run_paper_experiments.jl

# The script will automatically analyze results after experiments complete
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
- Small: 1-12 timesteps
- Medium: 1-26 timesteps  
- Large: 1-52 timesteps

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

All experiments use fixed random seeds for reproducibility. The seed is saved in `experiment_config.json` and can be reused to reproduce exact results.

## Tips

1. **Start Small**: Test with fewer simulations first to ensure everything works
2. **Monitor Progress**: The scripts show progress bars during execution
3. **Check Logs**: Verbose output includes policy generation times and convergence info
4. **Parallel Runs**: Consider running different problem sizes in parallel terminals
5. **Storage**: Full experiments generate ~1-2GB of data including plots