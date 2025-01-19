# README

This repository contains a Julia script that defines and simulates a Partially Observable Markov Decision Process (POMDP) for a project end-time estimation problem. The code demonstrates how to:
1. Define a POMDP model using the [QuickPOMDPs.jl](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) interface.
2. Solve the POMDP using different solvers (random, FIB, QMDP, SARSOP).
3. Simulate the resulting policies and collect rewards.
4. Plot the distribution of rewards across multiple simulations.

Below is a high-level overview of the different sections in the script.

---

## Table of Contents

1. [Dependencies](#dependencies)  
2. [Problem Description](#problem-description)  
3. [Project Structure](#project-structure)  
4. [Script Overview](#script-overview)  
5. [How to Run](#how-to-run)  
6. [Interpretation of Results](#interpretation-of-results)

---

## Dependencies

This script relies on several Julia packages, all of which can be installed using the Julia package manager. To ensure all dependencies are satisfied, open the Julia REPL and type:

```julia
using Pkg
Pkg.add([
    "POMDPs",
    "QuickPOMDPs",
    "POMDPPolicies",
    "POMDPSimulators",
    "POMDPModelTools",
    "QMDP",
    "NativeSARSOP",
    "Distributions",
    "POMDPTools",
    "FIB",
    "PyPlot",
    "Debugger",
    "ProgressMeter",
    "LinearAlgebra",
    "Random"
])
```

Below is a short summary of the roles each package plays:

- **POMDPs.jl**: Framework for defining POMDP problems in Julia.
- **QuickPOMDPs.jl**: Simplifies the definition of POMDP models.
- **POMDPPolicies.jl** and **POMDPSimulators.jl**: Tools for simulating policies in POMDPs.
- **QMDP.jl**, **FIB.jl**, **NativeSARSOP.jl**: Solvers for POMDPs (QMDP, FIB, and SARSOP respectively).
- **PyPlot.jl**: Used for plotting histograms of reward distributions.
- **ProgressMeter.jl**: Progress bar for simulation loops.
- **Debugger.jl**: Used for debugging (optional in normal usage).
- **Dates.jl**, **Random.jl**, **Distributions.jl**, **LinearAlgebra.jl**: Standard Julia libraries for dates, random generation, probability distributions, and linear algebra.

---

## Problem Description

The script models a scenario in which a project has an uncertain completion time. We want to estimate (or *announce*) when the project will finish based on partial observations of its progress.

- **States**: A state is a tuple \((t, Ta, Ts)\) where:
  - \(t\) is the current time step in the project (from 0 to the true end time).
  - \(Ta\) is the currently *announced* time for completion.
  - \(Ts\) is the *true* end time (unknown to the decision-maker, but modeled in the simulation).
  - The parameters `max_end_time`, `min_end_time`, `max_estimated_time`, and `min_estimated_time` define the range of possible true end times (`Ts`), current announced times (`Ta`), and the ongoing time steps (`t`). By modifying these global variables, you can easily shrink or expand the state space to reflect different project durations or estimate ranges for completion times.
    
- **Actions**:
  1. `:dont_announce`: Do not revise the announced time.
  2. `AnnounceAction(announced_time)`: Revise the announced time to `announced_time`.

- **Transition**: Moves forward in time and updates the state based on the chosen action. If the action is `AnnounceAction(...)`, the announced time (`Ta`) is updated accordingly.

- **Observations**: After each step, we receive an observation \((t, Ta, To)\), where `To` is a noisy estimate of the true end time \(Ts\). This noisy estimate is generated via a truncated normal distribution centered around \(Ts\) with a decreasing standard deviation as \(t\) approaches \(Ts\). (Observed times are restricted to times following the current time $t$.)

- **Reward**:
  - A negative reward is given proportional to the absolute difference \(|Ta - Ts|\).  
  - Large negative penalties occur if the announcement is infeasible (e.g., announcing a time that is already past or not matching the actual completion time once the project is finished).
  - Small negative cost for each step to encourage fewer announcements.

---

## Project Structure

Although this is a single script, conceptually it has several components:

1. **POMDP Definition**  
   - *`define_pomdp()`*: Creates a `QuickPOMDP` model with state space, action space, observations, transition, observation model, and reward function.

2. **Helpers**  
   - *`print_belief_states_and_probs(belief)`*: Utility to print out the belief distribution over states.

3. **Policy Computation**  
   - *`get_policy(pomdp, solver_type)`*: Chooses a solver, computes a policy, and measures the solver’s runtime.

4. **Simulation**  
   - *`simulate_single(pomdp, policy; verbose=true)`*: Simulates one episode, step by step, using a specific policy. 
   - *`simulate_many(pomdp, solver_type, num_simulations)`*: Runs multiple simulations and gathers statistics on rewards and runtime.
   - Adjust the `NUM_SIMULATIONS` global variable to change the number of simulations run.

5. **Visualization**  
   - *`plot_rewards(rewards, solver_type)`*: Plots a histogram of the rewards obtained across simulations.

6. **Main Entry Point**  
   - *`main()`*: Parses command-line arguments and:
     1. Creates the POMDP.
     2. Solves it using the chosen solver.
     3. Performs a single demonstration run (verbose).
     4. Runs multiple simulations and plots the distribution of rewards.

---

## Script Overview

```julia
# 1. Load dependencies
using Pkg
using Dates
using POMDPs
using QuickPOMDPs
...

# 2. Define a custom action type
struct AnnounceAction
    announced_time::Int
end

# 3. Define the POMDP model
function define_pomdp()
    ...
    return pomdp
end

# 4. Utility functions
function print_belief_states_and_probs(belief)
    ...
end

# 5. Compute policies using different solvers
function get_policy(pomdp, solver_type)
    ...
end

# 6. Simulation functions
function simulate_single(pomdp, policy; verbose=true)
    ...
end

function simulate_many(pomdp, solver_type, num_simulations)
    ...
end

function simulate(pomdp, solver_type)
    ...
end

# 7. Visualization
function plot_rewards(rewards, solver_type)
    ...
end

# 8. Main entry point
function main()
    ...
end

main()
```

---

## How to Run

1. **Clone or Download** this repository.

2. **Install the required packages** (see [Dependencies](#dependencies)).

3. **Run the script** from the command line, specifying one of the solvers:
   ```bash
   julia script_name.jl sarsop
   ```
   The valid solver types are:
   - `random`
   - `fib`
   - `qmdp`
   - `sarsop`

   If no valid solver type is passed, the script defaults to the random policy.

4. **View Outputs**:
   - The script will print a step-by-step simulation (for a single run).
   - Then it will simulate multiple runs and report average reward and iteration times.
   - A histogram of rewards is saved to `plots/histogram_<solver_type>.png`.
   - See `results.json` for comprehensive statistics of all runs. 

---

## Interpretation of Results

- **Single-run Output**:
  - Step-by-step transitions, actions, observations, and resulting rewards.  
  - Allows you to see how the belief (probability distribution over states) evolves over time.

- **Multiple-run Statistics**:

  - **Total reward** and **Average reward**: Higher rewards indicate better alignment of announcements (`Ta`) with the true end time (`Ts`).  
  - **Average iteration time**: Provides insight into computational performance.
    

- **Histogram**:
  - Depicts the distribution of total rewards from the runs. A narrower and higher distribution centered around higher rewards typically indicates a more consistent and accurate policy.
- **Results JSON**
  - See `results.json` for comprehensive statistics of all runs.
  - It has the following structure
    ```
    {
    "run_details": [
        [
            # details for each timestep of a run
            {
                "timestep": number,
                "Ts": number, # true end time
                "Ta_prev": number
                "To_prev": number | null,
                "action": {
                    "announced_time": number
                } | "string",
                "To": number,
                "reward": number,       
            },
            ...
        ],
        ...
    ],
    "iteration_times": [number, number, ...],
    "Ts_max": number,
    "To_min": number,
    "comp_policy_time": number,
    "rewards": [number, number, ...],
    "Ts_min": number,
    "solver_type": "string"
    }
    ```
- **Analysis**
  - `analysis.py` allows you to analyze the output from results.json

