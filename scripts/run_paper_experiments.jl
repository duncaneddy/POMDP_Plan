#!/usr/bin/env julia

# Script to run all experiments for the paper with memory-efficient incremental saving

include("analyze_paper_results.jl")

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using POMDPPlanning

# Configuration
SOLVERS = ["OBSERVEDTIME", "MOSTLIKELY", "QMDP", "MOMDP_SARSOP"]
POLICY_TIMEOUT = 60*30    # 30 minutes for policy computation
NUM_CONDITIONS = 100      # Number of distinct initial Tt values
NUM_REPETITIONS = 10      # Number of stochastic repetitions per condition
NUM_DETAILED_PLOTS = 25   # Number of runs to save detailed belief plots for
SAVE_FREQUENCY = 50       # Save results every N simulations to prevent memory growth
OUTPUT_DIR = "paper_results"
SEED = 42  # For reproducibility
VERBOSE = true  # Set to false for less output
LAMBDA_C = 2.0  # Quadratic accuracy penalty weight
LAMBDA_E = 8.0  # Change-magnitude penalty weight

"""
Load problem size configurations.
"""
function load_problem_configs()
    configs = Dict{String, Dict{String, Any}}(
        "small" => Dict{String, Any}(
            "min_end_time" => 2,
            "max_end_time" => 13
        ),
        "medium" => Dict{String, Any}(
            "min_end_time" => 2,
            "max_end_time" => 26
        ),
        "large" => Dict{String, Any}(
            "min_end_time" => 2,
            "max_end_time" => 39
        ),
        "xlarge" => Dict{String, Any}(
            "min_end_time" => 2,
            "max_end_time" => 52
        )
    )
    return configs
end

"""
Estimate memory usage and recommend save frequency.
"""
function recommend_save_frequency(num_simulations::Int, num_problem_sizes::Int, num_solvers::Int)
    detailed_sim_size_mb = 5.0
    consolidated_sim_size_kb = 1.0
    memory_budget_mb = 2000

    memory_per_batch_mb = (NUM_DETAILED_PLOTS * detailed_sim_size_mb) +
                         (SAVE_FREQUENCY * consolidated_sim_size_kb / 1000)

    if memory_per_batch_mb > memory_budget_mb
        recommended_freq = max(10, Int(floor(memory_budget_mb / detailed_sim_size_mb)))
        println("Warning: Large memory usage expected. Recommending save frequency: $recommended_freq")
        return recommended_freq
    end

    return SAVE_FREQUENCY
end

function main()
    println("="^60)
    println("Running Paper Experiments (Initial Conditions Version)")
    println("="^60)

    # Load problem configurations
    problem_configs = load_problem_configs()

    # Estimate memory usage and adjust save frequency if needed
    actual_save_freq = SAVE_FREQUENCY
    num_simulations = NUM_CONDITIONS * NUM_REPETITIONS

    println("\nConfiguration:")
    println("  Solvers: $(join(SOLVERS, ", "))")
    println("  Policy timeout: $POLICY_TIMEOUT seconds")
    println("  Initial conditions: $NUM_CONDITIONS")
    println("  Repetitions per condition: $NUM_REPETITIONS")
    println("  Total simulations per solver: $num_simulations")
    println("  Detailed plots: $NUM_DETAILED_PLOTS runs")
    println("  Save frequency: $actual_save_freq simulations")
    println("  Lambda_c: $LAMBDA_C")
    println("  Lambda_e: $LAMBDA_E")
    println("  Random seed: $SEED")
    println("  Output directory: $OUTPUT_DIR")
    println("  Verbose mode: $VERBOSE")
    println("  Problem configurations: $(keys(problem_configs))")
    println("  Total problems: $(length(problem_configs))")

    # Estimate total runtime
    estimated_policy_time = length(SOLVERS) * length(problem_configs) * POLICY_TIMEOUT / 60
    estimated_sim_time = length(SOLVERS) * length(problem_configs) * num_simulations * 0.1 / 60
    total_estimated_minutes = estimated_policy_time + estimated_sim_time

    println("  Estimated runtime: $(round(total_estimated_minutes, digits=1)) minutes")
    println()

    # Run experiments
    experiment_dir, results = POMDPPlanning.run_paper_experiments(
        problem_configs,
        SOLVERS,
        OUTPUT_DIR,
        num_conditions = NUM_CONDITIONS,
        num_repetitions = NUM_REPETITIONS,
        num_detailed_plots = NUM_DETAILED_PLOTS,
        policy_timeout = POLICY_TIMEOUT,
        seed = SEED,
        verbose = VERBOSE,
        save_frequency = actual_save_freq,
        lambda_c = LAMBDA_C,
        lambda_e = LAMBDA_E
    )
    
    println("\n" * "="^60)
    println("Experiments complete!")
    println("Results saved to: $experiment_dir")
    println("\nFile structure:")
    println("  $(experiment_dir)/")
    println("  ├── experiment_config.json              # Experiment configuration")
    println("  ├── initial_conditions_<size>.json      # Initial conditions per problem size")
    println("  ├── all_results.json                    # Consolidated results for analysis")
    println("  ├── results_<size>.json                 # Results by problem size")
    println("  ├── detailed_data/                      # Detailed simulation data")
    println("  │   └── <size>/<solver>/")
    println("  │       ├── consolidated_batch_*.json   # Batched consolidated metrics")
    println("  │       └── detailed_batch_*.json       # Batched detailed data")
    println("  └── belief_evolution_plots/             # Detailed belief evolution plots")
    println("="^60)
    
    # Print memory usage summary
    total_files = 0
    total_size_mb = 0.0
    
    if isdir(experiment_dir)
        for (root, dirs, files) in walkdir(experiment_dir)
            for file in files
                filepath = joinpath(root, file)
                if isfile(filepath)
                    total_files += 1
                    total_size_mb += stat(filepath).size / (1024 * 1024)
                end
            end
        end
    end
    
    println("\nDisk usage summary:")
    println("  Total files created: $total_files")
    println("  Total disk usage: $(round(total_size_mb, digits=1)) MB")
    println("  Average per simulation: $(round(total_size_mb / (num_simulations * length(SOLVERS) * length(problem_configs)), digits=3)) MB")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end