#!/usr/bin/env julia

# Script to run all experiments for the paper with memory-efficient incremental saving

include("analyze_paper_results.jl")

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using POMDPPlanning

# Configuration
SOLVERS = ["OBSERVEDTIME", "MOSTLIKELY", "QMDP", "CXX_SARSOP", "MOMDP_SARSOP"]
SOLVERS = ["OBSERVEDTIME", "MOSTLIKELY", "QMDP"]
POLICY_TIMEOUT = 60*5  # 30 minutes for policy computation
NUM_SIMULATIONS = 250  # Number of simulations per solver/problem
NUM_DETAILED_PLOTS = 25  # Number of runs to save detailed belief plots for
SAVE_FREQUENCY = 50     # Save results every N simulations to prevent memory growth
OUTPUT_DIR = "paper_results"
SEED = 42  # For reproducibility
VERBOSE = true  # Set to false for less output

"""
Load problem size configurations from reference problems.
"""
function load_problem_configs(reference_dir::String="reference_problems")
    configs = Dict{String, Dict{String, Any}}()
    
    # Define problem sizes
    problem_definitions = Dict(
        "small" => Dict(
            "filename" => "std_div_3/qmdp_base_l_2_u_12_n_1000.json",
            "min_end_time" => 2,
            "max_end_time" => 12
        ),
        # "medium" => Dict(
        #     "filename" => "std_div_3/qmdp_base_l_2_u_26_n_1000.json", 
        #     "min_end_time" => 2,
        #     "max_end_time" => 26
        # ),
        # "large" => Dict(
        #     "filename" => "std_div_3/qmdp_base_l_2_u_52_n_1000.json", 
        #     "min_end_time" => 2,
        #     "max_end_time" => 52
        # )
    )
    
    for (size_name, size_def) in problem_definitions
        filepath = joinpath(reference_dir, size_def["filename"])
        if !isfile(filepath)
            error("Reference file not found: $filepath")
        end
        
        configs[size_name] = Dict(
            "replay_data_path" => filepath,
            "min_end_time" => size_def["min_end_time"],
            "max_end_time" => size_def["max_end_time"]
        )
    end
    
    return configs
end

"""
Estimate memory usage and recommend save frequency.
"""
function recommend_save_frequency(num_simulations::Int, num_problem_sizes::Int, num_solvers::Int)
    # Rough estimates:
    # - Each simulation with detailed data: ~1-10 MB (belief history, iterations)
    # - Each simulation consolidated: ~1 KB
    # - Memory target: Keep under 4 GB in memory at once
    
    detailed_sim_size_mb = 5.0  # Conservative estimate per detailed simulation
    consolidated_sim_size_kb = 1.0  # Conservative estimate per consolidated simulation
    
    # Memory budget in MB
    memory_budget_mb = 2000  # 2 GB buffer
    
    # Calculate recommended frequency
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
    println("Running Paper Experiments (Memory-Efficient Version)")
    println("="^60)
    
    # Load problem configurations
    problem_configs = load_problem_configs("reference_problems")
    
    # Estimate memory usage and adjust save frequency if needed
    actual_save_freq = SAVE_FREQUENCY
    
    println("\nConfiguration:")
    println("  Solvers: $(join(SOLVERS, ", "))")
    println("  Policy timeout: $POLICY_TIMEOUT seconds")
    println("  Simulations per solver: $NUM_SIMULATIONS")
    println("  Detailed plots: $NUM_DETAILED_PLOTS runs")
    println("  Save frequency: $actual_save_freq simulations")
    println("  Random seed: $SEED")
    println("  Output directory: $OUTPUT_DIR")
    println("  Verbose mode: $VERBOSE")
    println("  Problem configurations: $(keys(problem_configs))")
    println("  Total problems: $(length(problem_configs))")
    
    # Estimate total runtime
    estimated_policy_time = length(SOLVERS) * length(problem_configs) * POLICY_TIMEOUT / 60  # Convert to minutes
    estimated_sim_time = length(SOLVERS) * length(problem_configs) * NUM_SIMULATIONS * 0.1 / 60  # Assume 0.1s per sim
    total_estimated_minutes = estimated_policy_time + estimated_sim_time
    
    println("  Estimated runtime: $(round(total_estimated_minutes, digits=1)) minutes")
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
        verbose = VERBOSE,
        save_frequency = actual_save_freq
    )
    
    println("\n" * "="^60)
    println("Experiments complete!")
    println("Results saved to: $experiment_dir")
    println("\nFile structure:")
    println("  $(experiment_dir)/")
    println("  ├── experiment_config.json       # Experiment configuration")
    println("  ├── all_results.json            # Consolidated results for analysis")
    println("  ├── results_<size>.json         # Results by problem size")
    println("  ├── detailed_data/              # Detailed simulation data")
    println("  │   └── <size>/<solver>/")
    println("  │       ├── consolidated_batch_*.json  # Batched consolidated metrics")
    println("  │       ├── detailed_batch_*.json      # Batched detailed data")
    println("  │       └── simulation_data.json       # Replay data")
    println("  └── belief_evolution_plots/     # Detailed belief evolution plots")
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
    println("  Average per simulation: $(round(total_size_mb / (NUM_SIMULATIONS * length(SOLVERS) * length(problem_configs)), digits=3)) MB")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end