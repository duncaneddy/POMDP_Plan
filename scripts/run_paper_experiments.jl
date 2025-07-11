#!/usr/bin/env julia

# Script to run all experiments for the paper

include("analyze_paper_results.jl")

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using POMDPPlanning

# Configuration
# SOLVERS = ["OBSERVEDTIME", "MOSTLIKELY", "QMDP", "MOMDP_SARSOP"]
SOLVERS = ["OBSERVEDTIME", "MOSTLIKELY"]
POLICY_TIMEOUT = 300  # 5 minutes for SARSOP solvers
NUM_SIMULATIONS = 100  # Number of simulations per solver/problem
NUM_DETAILED_PLOTS = 15  # Number of runs to save detailed belief plots for
OUTPUT_DIR = "paper_results"
SEED = 42  # For reproducibility
VERBOSE = true  # Set to false for less output

function main()
    println("="^60)
    println("Running Paper Experiments")
    println("="^60)
    
    # Load problem configurations
    problem_configs = POMDPPlanning.load_problem_configs("reference_problems")
    
    println("\nConfiguration:")
    println("  Solvers: $(join(SOLVERS, ", "))")
    println("  Policy timeout: $POLICY_TIMEOUT seconds")
    println("  Simulations per solver: $NUM_SIMULATIONS")
    println("  Detailed plots: $NUM_DETAILED_PLOTS runs")
    println("  Random seed: $SEED")
    println("  Output directory: $OUTPUT_DIR")
    println("  Verbose mode: $VERBOSE")
    println("  Problem configurations: $(keys(problem_configs))")
    println("  Total problems: $(length(problem_configs))")
    println()
    
    # Run experiments
    experiment_dir, results = POMDPPlanning.run_paper_experiments(
        problem_configs,
        SOLVERS,
        OUTPUT_DIR,
        num_simulations = NUM_SIMULATIONS,
        num_detailed_plots = NUM_DETAILED_PLOTS,
        policy_timeout = POLICY_TIMEOUT,
        seed = SEED,
        verbose = VERBOSE
    )
    
    println("\n" * "="^60)
    println("Experiments complete!")
    println("Results saved to: $experiment_dir")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end