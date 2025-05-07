# POMDPPlanning

A Julia framework for solving and simulating Partially Observable Markov Decision Process (POMDP) for project end-time estimation problems.

## 1. Problem Description

This package models a scenario where a project has an uncertain completion time. The goal is to accurately estimate (or *announce*) when the project will finish based on partial observations of its progress.

### Model Components

- **States**: Represented as tuples `(t, Ta, Tt)` where:
  - `t`: Current timestep in the project (from 0 to the true end time)
  - `Ta`: Currently *announced* completion time
  - `Tt`: *True* end time (unknown to the decision-maker)

- **Actions**: 
  - `AnnounceAction(announced_time)`: Announce/revise the estimated completion time

- **Transitions**: 
  - Time moves forward deterministically with each step
  - The announced time (`Ta`) is updated based on the chosen action

- **Observations**: 
  - After each step, a noisy estimate of the true end time is generated
  - These observations become more accurate as the project approaches completion
  - Observations follow a truncated normal distribution centered on the true end time

- **Reward**:
  - Penalties for inaccurate announcements (proportional to the difference from true end time)
  - Larger penalties for announcing impossible times (e.g., time already passed)
  - Different penalties for overestimating vs. underestimating completion time
  - Small ongoing costs to encourage fewer announcement changes

## 2. Setup / Installation


1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/POMDPPlanning.git
   cd POMDPPlanning
   ```

2. Install as a local package (optional):
   ```julia
   using Pkg
   Pkg.develop(path=".")
   ```

## 3. Running CLI

The package includes a command-line interface for solving POMDPs and evaluating policies.

### Solve a POMDP

```bash
julia bin/cli.jl solve --solvers SARSOP --min-end-time 10 --max-end-time 20 --discount 0.99
```

Options:
- `--solvers, -s`: Solver type (SARSOP, FIB, QMDP, PBVI, POMCPOW)
- `--min-end-time, -l`: Minimum possible end time
- `--max-end-time, -u`: Maximum possible end time
- `--discount, -d`: Discount factor (between 0 and 1)
- `--verbose, -v`: Enable verbose output
- `--debug, -D`: Enable debug output
- `--output-dir, -o`: Directory for saving results
- `--seed, -r`: Random seed for reproducibility

### Evaluate a Policy

```bash
julia bin/cli.jl evaluate --policy-file results/policy_sarsop.jld2 --num_simulations 10 -r 0
```

Options:
- `--policy-file, -p`: Path to the saved policy file (.jld2)
- `--true-end-time, -t`: Fixed true end time for evaluation
- `--initial-announce, -a`: Initial announced time
- Plus the options from the `solve` command

### Run an Experiment

Run an experiment comparing specific solvers (with fixed seed for reproducibility):

```bash
julia bin/cli.jl experiments --solvers=QMDP,SARSOP --min-end-time=10 --max-end-time=20 --num_simulations=5 --verbose -r 0
```

Run an experiment with all solvers:

```bash
julia bin/cli.jl experiments --solvers=all --min-end-time=10 --max-end-time=20 --num_simulations=5 --verbose
```

## 4. Project Structure

```
POMDPPlanning/
├── bin/
│   └── cli.jl                   # Command-line interface entry point
├── src/
│   ├── POMDPPlanning.jl         # Main module file
│   ├── problem.jl               # POMDP definition
│   ├── solvers.jl               # Solver implementations
│   ├── simulation.jl            # Policy simulation functions
│   ├── analysis.jl              # Evaluation and plotting functions
│   ├── utils.jl                 # Utility functions
│   └── experiments.jl           # Experiment definitions
├── Project.toml                 # Project configuration
└── README.md                    # This file
```

### Key Components

- **POMDPPlanning.jl**: Main module that integrates all components
- **problem.jl**: Defines the POMDP structure with state space, transition function, observation model, and reward function
- **solvers.jl**: Implements different solvers including FIB, PBVI, POMCPOW, QMDP, and SARSOP
- **simulation.jl**: Functions for single and batch simulations of policies
- **analysis.jl**: Metrics calculation and visualization tools

### Output Structure

```
output/
├── policy_sarsop.jld2           # Saved policy file
├── policy_sarsop.json           # Policy metadata (human-readable)
├── evaluation_results.json      # Detailed evaluation metrics
└── plots/                       # Generated visualizations
    ├── reward_distribution.png
    ├── error_metrics.png
    ├── num_changes.png
    └── undershoot_overshoot.png
```

The evaluation outputs include comprehensive metrics such as initial vs. final errors, number of announcement changes, and whether the final estimate was an undershoot or overshoot of the true end time.

## Analysis Capabilities

The framework provides tools for visualizing and analyzing policy performance:

1. **Reward distribution**: Histogram showing the performance across simulations
2. **Error metrics**: Visualization of initial errors, final errors, and change magnitudes
3. **Number of changes**: Distribution of how many times estimates were revised
4. **Undershoot vs. overshoot**: Analysis of whether policies tend to underestimate or overestimate completion times

These visualizations help in comparing different solvers and understanding their behavior in project end-time estimation tasks.