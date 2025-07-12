#!/usr/bin/env julia

# Script to run all experiments for the paper

include("analyze_paper_results.jl")

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using POMDPPlanning

# Configuration
SOLVERS = ["OBSERVEDTIME", "MOSTLIKELY", "QMDP", "MOMDP_SARSOP"]
POLICY_TIMEOUT = 60*30
NUM_SIMULATIONS = 1000  # Number of simulations per solver/problem
NUM_DETAILED_PLOTS = 20  # Number of runs to save detailed belief plots for
OUTPUT_DIR = "paper_results"
SEED = 42  # For reproducibility
VERBOSE = false  # Set to false for less output

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
        "medium" => Dict(
            "filename" => "std_div_3/qmdp_base_l_2_u_26_n_1000.json", 
            "min_end_time" => 2,
            "max_end_time" => 26
        ),
        "large" => Dict(
            "filename" => "std_div_3/qmdp_base_l_2_u_52_n_1000.json", 
            "min_end_time" => 2,
            "max_end_time" => 52
        )
        # "medium" => Dict(
        #     "filename" => "../results/evaluation_2025-07-12_01-11-56/evaluation_results.json", 
        #     "min_end_time" => 2,
        #     "max_end_time" => 12
        # )
        # "small" => Dict(
        #     "filename" => "std_div_3/problems_l_1_u_12_n_1000_s_42.json",
        #     "min_end_time" => 1,
        #     "max_end_time" => 12
        # ),
        # "medium" => Dict(
        #     "filename" => "std_div_3/problems_l_1_u_26_n_1000_s_42.json", 
        #     "min_end_time" => 1,
        #     "max_end_time" => 26
        # ),
        # "medium" => Dict(
        #     "filename" => "std_div_3/problems_l_1_u_52_n_1000.json", 
        #     "min_end_time" => 1,
        #     "max_end_time" => 26
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

function main()
    println("="^60)
    println("Running Paper Experiments")
    println("="^60)
    
    # Load problem configurations
    problem_configs = load_problem_configs("reference_problems")
    
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