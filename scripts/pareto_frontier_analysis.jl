#!/usr/bin/env julia

# Pareto Frontier Analysis for Reward Parameter Trade-offs
# Analyzes trade-offs between prediction accuracy and announcement changes
# by varying lambda_c and lambda_e parameters

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using POMDPPlanning
using JSON
using Statistics
using Plots
using StatsPlots
using Printf
using DataFrames
using CSV
using ArgParse
using Random

# Configuration constants
const OUTPUT_DIR = "pareto_analysis_results"
const DEFAULT_REFERENCE_PROBLEM = "reference_problems/std_div_3/qmdp_base_l_2_u_26_n_1000.json"
const DEFAULT_SOLVER = "QMDP"
const DEFAULT_NUM_SIMULATIONS = 100

# Parameter sweep configurations
const DEFAULT_LAMBDA_C_SWEEP = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
const DEFAULT_LAMBDA_E_SWEEP = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
const DEFAULT_LAMBDA_F = 1000.0
const DEFAULT_STD_DIVISOR = 3.0
const DEFAULT_DISCOUNT = 0.99

# Plot settings
const PLOT_SETTINGS = Dict(
    :titlefontsize => 16,
    :labelfontsize => 14,
    :tickfontsize => 12,
    :legendfontsize => 12,
    :guidefontsize => 14,
    :left_margin => 15Plots.mm,
    :bottom_margin => 12Plots.mm,
    :right_margin => 10Plots.mm,
    :top_margin => 8Plots.mm,
    :dpi => 300,
    :fontfamily => "Computer Modern"
)

function parse_commandline()
    s = ArgParseSettings(
        description = "Pareto Frontier Analysis for Reward Parameter Trade-offs",
        add_help = true,
    )
    
    @add_arg_table! s begin
        "--reference-data", "-r"
            help = "Path to reference simulation data JSON file"
            arg_type = String
            default = DEFAULT_REFERENCE_PROBLEM
        "--solver", "-s"
            help = "Solver to use for analysis"
            arg_type = String
            default = DEFAULT_SOLVER
        "--num-simulations", "-n"
            help = "Number of simulations to run for each parameter combination"
            arg_type = Int
            default = DEFAULT_NUM_SIMULATIONS
        "--output-dir", "-o"
            help = "Output directory for results and plots"
            arg_type = String
            default = OUTPUT_DIR
        "--lambda-c-range"
            help = "Comma-separated values for lambda_c sweep"
            arg_type = String
            default = join(DEFAULT_LAMBDA_C_SWEEP, ",")
        "--lambda-e-range"
            help = "Comma-separated values for lambda_e sweep"
            arg_type = String
            default = join(DEFAULT_LAMBDA_E_SWEEP, ",")
        "--lambda-f"
            help = "Fixed value for lambda_f (final error penalty)"
            arg_type = Float64
            default = DEFAULT_LAMBDA_F
        "--discount", "-d"
            help = "Discount factor"
            arg_type = Float64
            default = DEFAULT_DISCOUNT
        "--std-divisor"
            help = "Standard deviation divisor for observations"
            arg_type = Float64
            default = DEFAULT_STD_DIVISOR
        "--verbose", "-v"
            help = "Enable verbose output"
            action = :store_true
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 42
    end
    
    return parse_args(s)
end

"""
Error metric calculations for evaluating prediction quality.
"""
struct ErrorMetrics
    avg_weighted_error::Float64      # Average error weighted by timesteps
    avg_timesteps_with_error::Float64 # Fraction of timesteps with error
    final_error::Float64             # Final prediction error
    rms_error::Float64              # Root mean square error over time
    max_error::Float64              # Maximum error at any timestep
    avg_absolute_error::Float64     # Simple average absolute error
end

function compute_error_metrics(run_details)
    """Compute comprehensive error metrics from simulation run details."""
    
    if isempty(run_details)
        return ErrorMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
    
    # Extract error at each timestep
    errors = [abs(step["action"] - step["Tt"]) for step in run_details]
    timesteps_with_error = sum(errors .> 0)
    total_timesteps = length(errors)
    
    # Compute metrics
    avg_weighted_error = sum(errors) / total_timesteps
    avg_timesteps_with_error = timesteps_with_error / total_timesteps
    final_error = errors[end]
    rms_error = sqrt(sum(errors.^2) / total_timesteps)
    max_error = maximum(errors)
    avg_absolute_error = mean(errors)
    
    return ErrorMetrics(
        avg_weighted_error,
        avg_timesteps_with_error, 
        final_error,
        rms_error,
        max_error,
        avg_absolute_error
    )
end

function aggregate_error_metrics(metrics_list::Vector{ErrorMetrics})
    """Aggregate error metrics across multiple simulation runs."""
    
    if isempty(metrics_list)
        return ErrorMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
    
    return ErrorMetrics(
        mean([m.avg_weighted_error for m in metrics_list]),
        mean([m.avg_timesteps_with_error for m in metrics_list]),
        mean([m.final_error for m in metrics_list]),
        mean([m.rms_error for m in metrics_list]),
        mean([m.max_error for m in metrics_list]),
        mean([m.avg_absolute_error for m in metrics_list])
    )
end

function extract_problem_parameters(reference_data_path::String)
    """Extract min/max end times from reference data path or content."""
    
    # Try to parse from filename first (more reliable)
    if contains(reference_data_path, "l_2_u_13")
        return 2, 13
    elseif contains(reference_data_path, "l_2_u_26")
        return 2, 26
    elseif contains(reference_data_path, "l_2_u_39")
        return 2, 39
    elseif contains(reference_data_path, "l_2_u_52")
        return 2, 52
    end
    
    # Fallback: try to infer from data content
    try
        reference_data = JSON.parsefile(reference_data_path)
        simulation_data = reference_data["simulation_data"]
        
        if !isempty(simulation_data)
            # Extract from initial states
            initial_states = [sim["initial_state"] for sim in simulation_data[1:min(10, length(simulation_data))]]
            true_end_times = [state[3] for state in initial_states if length(state) >= 3]
            
            if !isempty(true_end_times)
                min_tt = minimum(true_end_times)
                max_tt = maximum(true_end_times)
                return min_tt, max_tt
            end
        end
    catch e
        println("Warning: Could not extract parameters from data: $e")
    end
    
    # Final fallback
    println("Warning: Using default problem parameters (10, 20)")
    return 10, 20
end

function evaluate_reward_parameters(
    reference_data_path::String,
    solver::String,
    lambda_c::Float64,
    lambda_e::Float64,
    lambda_f::Float64,
    num_simulations::Int,
    discount::Float64,
    std_divisor::Float64,
    verbose::Bool
)
    """Evaluate a single reward parameter configuration."""
    
    if verbose
        println("Evaluating λc=$(lambda_c), λe=$(lambda_e), λf=$(lambda_f)")
    end
    
    # Load reference data
    reference_data = JSON.parsefile(reference_data_path)
    simulation_data = reference_data["simulation_data"]
    
    # Extract problem parameters
    min_end_time, max_end_time = extract_problem_parameters(reference_data_path)
    
    # Create POMDP with custom reward parameters
    pomdp = POMDPPlanning.define_pomdp(
        min_end_time,
        max_end_time,
        discount,
        std_divisor=std_divisor,
        lambda_c=lambda_c,
        lambda_e=lambda_e,
        lambda_f=lambda_f
    )
    
    # Generate policy for this parameter configuration
    policy_data = POMDPPlanning.get_policy(pomdp, solver, tempdir(), verbose=false)
    policy = policy_data["policy"]
    
    # Run simulations with subset of reference data
    actual_num_sims = min(num_simulations, length(simulation_data))
    replay_data = simulation_data[1:actual_num_sims]
    
    # Evaluate policy
    stats = POMDPPlanning.simulate_many(
        pomdp,
        policy,
        actual_num_sims,
        replay_data=replay_data,
        collect_beliefs=false,
        verbose=false
    )
    
    # Compute error metrics for each simulation
    error_metrics_list = ErrorMetrics[]
    for run_details in stats["run_details"]
        metrics = compute_error_metrics(run_details)
        push!(error_metrics_list, metrics)
    end
    
    # Aggregate metrics
    aggregated_metrics = aggregate_error_metrics(error_metrics_list)
    
    # Calculate announcement changes
    avg_changes = mean(stats["num_changes"])
    
    return Dict(
        "lambda_c" => lambda_c,
        "lambda_e" => lambda_e,
        "lambda_f" => lambda_f,
        "avg_changes" => avg_changes,
        "error_metrics" => aggregated_metrics,
        "total_reward" => mean(stats["rewards"]),
        "policy_solve_time" => policy_data["policy_solve_time"]
    )
end

function run_pareto_sweep(
    reference_data_path::String,
    solver::String,
    lambda_c_values::Vector{Float64},
    lambda_e_values::Vector{Float64},
    lambda_f::Float64,
    num_simulations::Int,
    discount::Float64,
    std_divisor::Float64,
    output_dir::String,
    verbose::Bool
)
    """Run comprehensive parameter sweep for Pareto frontier analysis."""
    
    mkpath(output_dir)
    
    # Three types of sweeps:
    # 1. Vary lambda_c, keep lambda_e constant
    # 2. Vary lambda_e, keep lambda_c constant  
    # 3. Vary ratio lambda_c/lambda_e, keep sum constant
    
    all_results = []
    
    println("Starting Pareto frontier analysis...")
    println("Parameter sweeps:")
    
    # Sweep 1: Vary lambda_c, keep lambda_e = 2.0 (middle value)
    base_lambda_e = 2.0
    println("  1. lambda_c sweep (λe fixed at $(base_lambda_e))")
    for lambda_c in lambda_c_values
        result = evaluate_reward_parameters(
            reference_data_path, solver, lambda_c, base_lambda_e, lambda_f,
            num_simulations, discount, std_divisor, verbose
        )
        result["sweep_type"] = "lambda_c_sweep"
        push!(all_results, result)
    end
    
    # Sweep 2: Vary lambda_e, keep lambda_c = 3.0 (middle value)
    base_lambda_c = 3.0
    println("  2. lambda_e sweep (λc fixed at $(base_lambda_c))")
    for lambda_e in lambda_e_values
        result = evaluate_reward_parameters(
            reference_data_path, solver, base_lambda_c, lambda_e, lambda_f,
            num_simulations, discount, std_divisor, verbose
        )
        result["sweep_type"] = "lambda_e_sweep"
        push!(all_results, result)
    end
    
    # Sweep 3: Vary ratio, keep sum constant
    lambda_sum = 5.0  # lambda_c + lambda_e = constant
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    println("  3. Ratio sweep (λc + λe = $(lambda_sum))")
    for ratio in ratios
        lambda_c = ratio * lambda_sum
        lambda_e = (1 - ratio) * lambda_sum
        result = evaluate_reward_parameters(
            reference_data_path, solver, lambda_c, lambda_e, lambda_f,
            num_simulations, discount, std_divisor, verbose
        )
        result["sweep_type"] = "ratio_sweep"
        result["lambda_ratio"] = ratio
        push!(all_results, result)
    end
    
    # Save all results
    results_path = joinpath(output_dir, "pareto_sweep_results.json")
    open(results_path, "w") do f
        JSON.print(f, all_results, 4)
    end
    
    println("Results saved to: $results_path")
    return all_results
end

function create_pareto_plots(results, output_dir::String)
    """Generate comprehensive Pareto frontier plots."""
    
    plots_dir = joinpath(output_dir, "plots")
    mkpath(plots_dir)
    
    # Set backend
    gr()
    
    # Group results by sweep type
    lambda_c_results = filter(r -> r["sweep_type"] == "lambda_c_sweep", results)
    lambda_e_results = filter(r -> r["sweep_type"] == "lambda_e_sweep", results)  
    ratio_results = filter(r -> r["sweep_type"] == "ratio_sweep", results)
    
    println("Generating Pareto frontier plots...")
    
    # Define different error metrics to plot
    error_metric_configs = [
        ("avg_weighted_error", "Average Weighted Error", "Error (Announcement vs True Time)"),
        ("avg_timesteps_with_error", "Fraction Timesteps with Error", "Fraction of Timesteps with Error"),
        ("final_error", "Final Prediction Error", "Final Error (Time Units)"),
        ("rms_error", "RMS Error", "RMS Error (Time Units)"),
        ("max_error", "Maximum Error", "Maximum Error (Time Units)")
    ]
    
    for (metric_key, metric_name, y_label) in error_metric_configs
        println("  Creating plots for: $metric_name")
        
        # Create combined plot with all three sweeps
        p_combined = plot(
            xlabel = "Average Number of Announcement Changes",
            ylabel = y_label,
            title = "Pareto Frontier: $metric_name vs Changes",
            legend = :topright,
            size = (1000, 700);
            PLOT_SETTINGS...
        )
        
        # Plot lambda_c sweep
        if !isempty(lambda_c_results)
            x_data_c = [r["avg_changes"] for r in lambda_c_results]
            y_data_c = [getfield(r["error_metrics"], Symbol(metric_key)) for r in lambda_c_results]
            
            plot!(p_combined, x_data_c, y_data_c,
                  label = "λc sweep (λe=2.0)",
                  marker = :circle,
                  markersize = 6,
                  linewidth = 2,
                  color = :blue)
        end
        
        # Plot lambda_e sweep  
        if !isempty(lambda_e_results)
            x_data_e = [r["avg_changes"] for r in lambda_e_results]
            y_data_e = [getfield(r["error_metrics"], Symbol(metric_key)) for r in lambda_e_results]
            
            plot!(p_combined, x_data_e, y_data_e,
                  label = "λe sweep (λc=3.0)", 
                  marker = :square,
                  markersize = 6,
                  linewidth = 2,
                  color = :red)
        end
        
        # Plot ratio sweep
        if !isempty(ratio_results)
            x_data_r = [r["avg_changes"] for r in ratio_results]
            y_data_r = [getfield(r["error_metrics"], Symbol(metric_key)) for r in ratio_results]
            
            plot!(p_combined, x_data_r, y_data_r,
                  label = "Ratio sweep (λc+λe=5.0)",
                  marker = :diamond,
                  markersize = 6,
                  linewidth = 2,
                  color = :green)
        end
        
        # Save in multiple formats
        base_filename = "pareto_$(metric_key)"
        savefig(p_combined, joinpath(plots_dir, "$(base_filename).pdf"))
        savefig(p_combined, joinpath(plots_dir, "$(base_filename).png"))
        savefig(p_combined, joinpath(plots_dir, "$(base_filename).svg"))
        
        # Create individual sweep plots
        for (sweep_results, sweep_name, color) in [
            (lambda_c_results, "lambda_c", :blue),
            (lambda_e_results, "lambda_e", :red),
            (ratio_results, "ratio", :green)
        ]
            if !isempty(sweep_results)
                p_individual = plot(
                    xlabel = "Average Number of Announcement Changes",
                    ylabel = y_label,
                    title = "$(metric_name) vs Changes ($(replace(sweep_name, "_" => " ")) sweep)",
                    legend = false,
                    size = (800, 600);
                    PLOT_SETTINGS...
                )
                
                x_data = [r["avg_changes"] for r in sweep_results]
                y_data = [getfield(r["error_metrics"], Symbol(metric_key)) for r in sweep_results]
                
                plot!(p_individual, x_data, y_data,
                      marker = :circle,
                      markersize = 8,
                      linewidth = 3,
                      color = color)
                
                # Add parameter labels
                if sweep_name == "lambda_c"
                    labels = [@sprintf("λc=%.1f", r["lambda_c"]) for r in sweep_results]
                elseif sweep_name == "lambda_e"
                    labels = [@sprintf("λe=%.1f", r["lambda_e"]) for r in sweep_results]
                else
                    labels = [@sprintf("r=%.1f", r["lambda_ratio"]) for r in sweep_results]
                end
                
                for (i, label) in enumerate(labels)
                    annotate!(p_individual, x_data[i], y_data[i],
                             text(label, 9, :bottom, :left))
                end
                
                # Save individual plots
                individual_filename = "pareto_$(metric_key)_$(sweep_name)_sweep"
                savefig(p_individual, joinpath(plots_dir, "$(individual_filename).pdf"))
                savefig(p_individual, joinpath(plots_dir, "$(individual_filename).png"))
                savefig(p_individual, joinpath(plots_dir, "$(individual_filename).svg"))
            end
        end
    end
    
    # Create summary statistics table
    create_pareto_summary_table(results, output_dir)
    
    println("Plots saved to: $plots_dir")
end

function create_pareto_summary_table(results, output_dir::String)
    """Create summary table of all parameter combinations and their performance."""
    
    # Convert results to DataFrame for easier analysis
    table_data = []
    
    for result in results
        metrics = result["error_metrics"]
        push!(table_data, Dict(
            "sweep_type" => result["sweep_type"],
            "lambda_c" => result["lambda_c"],
            "lambda_e" => result["lambda_e"],
            "lambda_ratio" => get(result, "lambda_ratio", missing),
            "avg_changes" => result["avg_changes"],
            "avg_weighted_error" => metrics.avg_weighted_error,
            "avg_timesteps_with_error" => metrics.avg_timesteps_with_error,
            "final_error" => metrics.final_error,
            "rms_error" => metrics.rms_error,
            "max_error" => metrics.max_error,
            "total_reward" => result["total_reward"],
            "policy_solve_time" => result["policy_solve_time"]
        ))
    end
    
    df = DataFrame(table_data)
    
    # Save as CSV
    CSV.write(joinpath(output_dir, "pareto_summary_table.csv"), df)
    
    # Find Pareto optimal points for each error metric
    error_metrics = ["avg_weighted_error", "avg_timesteps_with_error", "final_error", "rms_error", "max_error"]
    
    pareto_analysis = Dict()
    for metric in error_metrics
        # Find points that minimize error for each level of changes
        pareto_points = find_pareto_optimal_points(df, "avg_changes", metric)
        pareto_analysis[metric] = pareto_points
    end
    
    # Save Pareto analysis
    pareto_path = joinpath(output_dir, "pareto_optimal_points.json")
    open(pareto_path, "w") do f
        JSON.print(f, pareto_analysis, 4)
    end
    
    println("Summary table saved to: $(joinpath(output_dir, "pareto_summary_table.csv"))")
    println("Pareto optimal points saved to: $pareto_path")
end

function find_pareto_optimal_points(df, x_col, y_col)
    """Find Pareto optimal points that minimize y for each x value."""
    
    # Sort by x column first
    sorted_df = sort(df, x_col)
    
    pareto_points = []
    min_y_so_far = Inf
    
    for row in eachrow(sorted_df)
        if row[y_col] < min_y_so_far
            min_y_so_far = row[y_col]
            push!(pareto_points, Dict(pairs(row)))
        end
    end
    
    return pareto_points
end

function main()
    args = parse_commandline()
    
    # Set random seed
    Random.seed!(args["seed"])
    
    # Parse parameter ranges
    lambda_c_values = parse.(Float64, split(args["lambda-c-range"], ","))
    lambda_e_values = parse.(Float64, split(args["lambda-e-range"], ","))
    
    println("Pareto Frontier Analysis")
    println("========================")
    println("Reference data: $(args["reference-data"])")
    println("Solver: $(args["solver"])")
    println("Output directory: $(args["output-dir"])")
    println("Number of simulations per parameter: $(args["num-simulations"])")
    println("Lambda_c values: $lambda_c_values")
    println("Lambda_e values: $lambda_e_values")
    println("Lambda_f (fixed): $(args["lambda-f"])")
    println()
    
    # Check if reference data exists
    if !isfile(args["reference-data"])
        error("Reference data file not found: $(args["reference-data"])")
    end
    
    # Run parameter sweep
    results = run_pareto_sweep(
        args["reference-data"],
        args["solver"],
        lambda_c_values,
        lambda_e_values,
        args["lambda-f"],
        args["num-simulations"],
        args["discount"],
        args["std-divisor"],
        args["output-dir"],
        args["verbose"]
    )
    
    # Generate plots
    create_pareto_plots(results, args["output-dir"])
    
    println("\nPareto frontier analysis complete!")
    println("Results and plots saved to: $(args["output-dir"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end