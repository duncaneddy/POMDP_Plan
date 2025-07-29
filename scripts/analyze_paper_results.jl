#!/usr/bin/env julia

# Generate all plots and tables for the paper - works with incremental data structure
# IMPROVED VERSION with better formatting, consistent colors, and higher quality outputs

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using POMDPPlanning
using JSON
using Statistics
using Plots
using StatsPlots
using PlotlyJS
using Printf
using DataFrames
using CSV

const REGEN_BELIEF_HISTORIES = true  # Set to false to skip belief history regeneration

# Define consistent problem size ordering and color scheme
const PROBLEM_SIZE_ORDER = ["small", "medium", "large", "xlarge"]
const SOLVER_ORDER = ["OBSERVEDTIME", "MOSTLIKELY", "QMDP", "CXX_SARSOP", "MOMDP_SARSOP"]
const SOLVER_COLORS = Dict(
    "Observed Time" => :blue,
    "Most Likely" => :red, 
    "QMDP" => :green,
    "MOMDP SARSOP" => :purple,
    "SARSOP" => :orange
)

# Global plot settings for consistent formatting
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

"""
Format solver names for display (capitalize words, remove underscores).
"""
function format_solver_name(solver::String)
    solver_display_names = Dict(
        "OBSERVEDTIME" => "Observed Time",
        "MOSTLIKELY" => "Most Likely", 
        "QMDP" => "QMDP",
        "CXX_SARSOP" => "SARSOP",
        "MOMDP_SARSOP" => "MOMDP SARSOP",
        "SARSOP" => "SARSOP",
        "POMCPOW" => "POMCPOW",
        "FIB" => "FIB",
        "PBVI" => "PBVI"
    )
    
    return get(solver_display_names, solver, solver)
end

"""
Format problem size names for display.
"""
function format_problem_size(size::String)
    size_display_names = Dict(
        "small" => "Small",
        "medium" => "Medium",
        "large" => "Large", 
        "xlarge" => "XLarge"
    )
    
    return get(size_display_names, size, titlecase(size))
end

"""
Get consistent color for a solver across all plots.
"""
function get_solver_color(solver::String)
    # Convert original solver name to formatted name for color lookup
    formatted_name = format_solver_name(solver)
    return get(SOLVER_COLORS, formatted_name, :black)
end

"""
Sort problem sizes in the correct order.
"""
function sort_problem_sizes(sizes::Vector{String})
    # Filter to only include sizes that exist in our data
    existing_sizes = filter(s -> s in sizes, PROBLEM_SIZE_ORDER)
    # Add any unexpected sizes at the end
    unexpected_sizes = filter(s -> !(s in PROBLEM_SIZE_ORDER), sizes)

    return vcat(existing_sizes, sort(unexpected_sizes))
end

"""
Sort solvers in the correct order for consistent presentation.
"""
function sort_solvers(solvers::Vector{Any})
    # Filter to only include solvers that exist in our data
    existing_solvers = filter(s -> s in solvers, SOLVER_ORDER)
    # Add any unexpected solvers at the end
    unexpected_solvers = filter(s -> !(s in SOLVER_ORDER), solvers)
    return vcat(existing_solvers, sort(unexpected_solvers))
end

"""
Load results from the new incremental data structure.
"""
function load_incremental_results(experiment_dir::String)
    # Try to load consolidated results first
    all_results_path = joinpath(experiment_dir, "all_results.json")
    if isfile(all_results_path)
        return JSON.parsefile(all_results_path)
    end
    
    # If consolidated file doesn't exist, build from individual size files
    all_results = Dict()
    
    # Find all results files
    for file in readdir(experiment_dir)
        if startswith(file, "results_") && endswith(file, ".json")
            size_name = replace(file, "results_" => "", ".json" => "")
            size_results_path = joinpath(experiment_dir, file)
            size_results = JSON.parsefile(size_results_path)
            all_results[size_name] = size_results
        end
    end
    
    # If still no results, try to reconstruct from detailed_data directory
    if isempty(all_results)
        all_results = reconstruct_from_detailed_data(experiment_dir)
    end
    
    return all_results
end

"""
Extract belief states and probabilities - handles both original and deserialized beliefs.
"""
function extract_belief_states_and_probs_local(belief)
    if belief === nothing
        return [], []
    end
    
    # Handle our custom deserialized belief format
    if isa(belief, Dict) && haskey(belief, :is_sparse_cat)
        return belief[:states], belief[:probs]
    end
    
    # Handle original SparseCat format
    if isa(belief, POMDPTools.POMDPDistributions.SparseCat)
        return belief.vals, belief.probs
    end
    
    # Handle tuple format (single state)
    if isa(belief, Tuple)
        return [belief], [1.0]
    end
    
    # Fallback to original function if available
    try
        return POMDPPlanning.extract_belief_states_and_probs(belief)
    catch
        println("Warning: Could not extract belief states and probabilities")
        return [], []
    end
end

function deserialize_belief_history(serialized_beliefs, is_momdp::Bool=false)
    if serialized_beliefs === nothing
        return nothing
    end
    
    beliefs = []
    
    for serialized_belief in serialized_beliefs
        if serialized_belief === nothing
            push!(beliefs, nothing)
            continue
        end
        
        if haskey(serialized_belief, "error")
            # Skip beliefs that failed to serialize
            push!(beliefs, nothing)
            continue
        end
        
        try
            states = serialized_belief["states"]
            probs = serialized_belief["probabilities"]
            
            # Convert states back to tuples if needed
            converted_states = []
            for state in states
                if isa(state, Array)
                    if is_momdp && length(state) == 1
                        # For MOMDP, states are just integers
                        push!(converted_states, state[1])
                    else
                        # For POMDP, states are tuples
                        push!(converted_states, Tuple(state))
                    end
                else
                    push!(converted_states, state)
                end
            end
            
            # Ensure probabilities are normalized
            prob_sum = sum(probs)
            if prob_sum > 0
                normalized_probs = probs ./ prob_sum
            else
                normalized_probs = probs
            end
            
            # Create a custom belief object that works with extract_belief_states_and_probs
            belief_dict = Dict(
                :states => converted_states,
                :probs => normalized_probs,
                :is_sparse_cat => true
            )
            
            push!(beliefs, belief_dict)
        catch e
            println("Warning: Failed to deserialize belief: $e")
            # If reconstruction fails, use a placeholder
            push!(beliefs, nothing)
        end
    end
    
    return beliefs
end

"""
Generate belief evolution plots from experiment results JSON data.
"""
function generate_belief_evolution_plots_from_json(experiment_dir::String, output_dir::String)
    println("Generating belief evolution plots from JSON data...")
    
    # Create belief plots directory under analysis
    belief_plots_dir = joinpath(output_dir, "belief_evolution_plots")
    mkpath(belief_plots_dir)
    
    # Load detailed data from JSON files
    detailed_data_dir = joinpath(experiment_dir, "detailed_data")
    if !isdir(detailed_data_dir)
        println("Warning: No detailed_data directory found. Skipping belief evolution plots.")
        return
    end
    
    # Load problem configurations to determine POMDP vs MOMDP
    config_path = joinpath(experiment_dir, "experiment_config.json")
    config = JSON.parsefile(config_path)
    problem_sizes = config["problem_sizes"]
    
    # Process each problem size
    for (size_name, size_config) in problem_sizes
        size_plots_dir = joinpath(belief_plots_dir, size_name)
        mkpath(size_plots_dir)
        
        size_data_dir = joinpath(detailed_data_dir, size_name)
        if !isdir(size_data_dir)
            continue
        end
        
        # Get POMDP parameters for this size
        min_end_time = size_config["min_end_time"]
        max_end_time = size_config["max_end_time"]
        
        # Process each solver
        for solver_name in readdir(size_data_dir)
            solver_data_dir = joinpath(size_data_dir, solver_name)
            if !isdir(solver_data_dir)
                continue
            end
            
            solver_plots_dir = joinpath(size_plots_dir, solver_name)
            mkpath(solver_plots_dir)
            
            # Determine if this is MOMDP
            is_momdp = uppercase(solver_name) == "MOMDP_SARSOP"
            
            # Find detailed batch files
            batch_files = filter(f -> startswith(f, "detailed_batch"), readdir(solver_data_dir))
            
            plot_count = 0
            max_plots = 999  # Limit number of plots per solver
            
            for batch_file in batch_files
                if plot_count >= max_plots
                    break
                end
                
                batch_path = joinpath(solver_data_dir, batch_file)
                try
                    batch_data = JSON.parsefile(batch_path)
                    
                    for detailed_metrics in batch_data
                        if plot_count >= max_plots
                            break
                        end
                        
                        # Extract belief history and run details
                        serialized_belief_history = get(detailed_metrics, "belief_history", nothing)
                        
                        if serialized_belief_history === nothing
                            println("Warning: No belief_history found for $(solver_name) run $(detailed_metrics["simulation_id"])")
                            continue
                        end
                        
                        belief_history = deserialize_belief_history(serialized_belief_history, is_momdp)
                        run_details = detailed_metrics["iterations"]
                        
                        if belief_history === nothing || isempty(belief_history)
                            println("Warning: Belief history is empty after deserialization for $(solver_name) run $(detailed_metrics["simulation_id"])")
                            continue
                        end
                        
                        true_end_time = run_details[1]["Tt"]
                        
                        try
                            # Create 2D belief evolution plot with actions
                            p = plot_2d_belief_evolution_with_actions(
                                belief_history,
                                run_details,
                                true_end_time,
                                min_end_time,
                                max_end_time,
                                title_prefix="$(format_solver_name(solver_name)) - Run $(detailed_metrics["simulation_id"]) - ",
                                is_momdp=is_momdp,
                                show_legend=(uppercase(solver_name) == "OBSERVEDTIME")  # Only show legend for OBSERVEDTIME
                            )
                            
                            if p !== nothing
                                # Generate plots for png, pdf, and svg formats
                                filename = "belief_evolution_run_$(lpad(detailed_metrics["simulation_id"], 3, '0'))"
                                Plots.savefig(p, joinpath(solver_plots_dir, "$filename.pdf"))
                                plot_count += 1  # Only increment on successful plot creation
                            end
                        catch plot_error
                            println("Warning: Could not create belief plot for $(solver_name) run $(detailed_metrics["simulation_id"]): $plot_error")
                            continue
                        end
                    end
                catch batch_error
                    println("Warning: Could not load detailed batch file $batch_file for $solver_name: $batch_error")
                    continue
                end
            end
            
            if plot_count > 0
                println("  Generated $plot_count belief evolution plots for $solver_name ($size_name)")
            else
                println("  No belief evolution plots generated for $solver_name ($size_name)")
            end
        end
    end
end

"""
Enhanced 2D belief evolution plot with actions and optional legend control.
"""
function plot_2d_belief_evolution_with_actions(belief_history, run_details, true_end_time, min_end_time, max_end_time; title_prefix="", is_momdp=false, show_legend=true)
    # Create the base 2D belief evolution plot
    p = plot_2d_belief_evolution(belief_history, true_end_time, min_end_time, max_end_time, title_prefix=title_prefix, is_momdp=is_momdp)
    
    if p === nothing
        return nothing
    end
    
    # Extract timesteps and announced times from run details
    timesteps = [step["timestep"] for step in run_details]
    announced_times = [step["action"] for step in run_details]
    
    # Overlay the announced time trajectory
    Plots.plot!(p,
        timesteps[2:end],  # Skip the first timestep for better alignment
        announced_times[2:end],  # Skip the first action for better alignment
        label = show_legend ? "Announced Time" : nothing,
        color = :white,
        linewidth = 3,
        marker = :circle,
        markersize = 4,
        markerstrokecolor = :black,
        markerstrokewidth = 1
    )
    
    # Also overlay observations if available
    if haskey(run_details[1], "To")
        observations = [step["To"] for step in run_details]
        Plots.scatter!(p,
            timesteps,
            observations,
            label = show_legend ? "Observations" : nothing,
            color = :yellow,
            marker = :diamond,
            markersize = 5,
            markerstrokecolor = :black,
            markerstrokewidth = 1
        )
    end
    
    # Control legend display
    if !show_legend
        Plots.plot!(p, legend = false)
    else
        Plots.plot!(p, legend = :topleft)
    end
    
    return p
end

"""
Generate 2D belief evolution heatmap.
"""
function plot_2d_belief_evolution(belief_history, true_end_time, min_end_time, max_end_time; title_prefix="", is_momdp=false)
    if belief_history === nothing || isempty(belief_history)
        @warn "No belief history available for 2D belief evolution plot"
        return nothing
    end
    
    # Ensure we're using the GR backend for heatmaps
    gr()
    
    num_timesteps = length(belief_history)
    possible_end_times = collect(min_end_time:max_end_time)
    num_end_times = length(possible_end_times)
    
    # Initialize probability matrix: rows = end times, columns = timesteps
    prob_matrix = zeros(Float64, num_end_times, num_timesteps)
    
    # Fill the probability matrix
    for (timestep_idx, belief) in enumerate(belief_history)
        if belief === nothing
            continue
        end
        
        states, probs = extract_belief_states_and_probs_local(belief)
        
        if isempty(states) || isempty(probs)
            continue
        end
        
        # Aggregate probabilities by true end time (Tt)
        end_time_probs = Dict{Int, Float64}()
        for (state, prob) in zip(states, probs)
            if is_momdp
                Tt = state  # For MOMDP, state is just the true end time
            else
                # For standard POMDP, state is a tuple (t, Ta, Tt)
                if isa(state, Tuple) && length(state) >= 3
                    Tt = state[3]
                else
                    println("Warning: Unexpected state format: $state")
                    continue
                end
            end
            
            # Only include end times within our range
            if Tt >= min_end_time && Tt <= max_end_time
                end_time_probs[Tt] = get(end_time_probs, Tt, 0.0) + prob
            end
        end
        
        # Fill the matrix column for this timestep
        for (end_time_idx, end_time) in enumerate(possible_end_times)
            prob_matrix[end_time_idx, timestep_idx] = get(end_time_probs, end_time, 0.0)
        end
    end
    
    # If matrix is all zeros or uniform, there's an issue
    if maximum(prob_matrix) == 0.0
        println("Warning: Probability matrix is all zeros - belief history may not be properly deserialized")
        return nothing
    end
    
    # Create timestep labels (starting from 0)
    timestep_labels = collect(0:(num_timesteps-1))
    
    # Create the heatmap using Plots.heatmap explicitly
    p = Plots.heatmap(
        timestep_labels,
        possible_end_times,
        prob_matrix,
        xlabel = "Simulation Time (t)",
        ylabel = "True End Time (Tt)",
        color = :viridis,
        aspect_ratio = :auto,
        size = (800, 600),
        fontfamily = "Computer Modern",
        colorbar_title = "Probability";
        PLOT_SETTINGS...
    )
    
    # Add a horizontal line for the actual true end time
    Plots.hline!(p, [true_end_time], 
           label = "True End Time", 
           color = :red, 
           linewidth = 3, 
           linestyle = :dash)

    # Ensure proper tick spacing for readability
    Plots.plot!(p,
        xticks = (0:2:maximum(timestep_labels), 0:2:maximum(timestep_labels)),
        yticks = (min_end_time:2:max_end_time, min_end_time:2:max_end_time)
    )
    
    return p
end

"""
Create LaTeX formatted reward table.
"""
function create_latex_reward_table(df::DataFrame, filepath::String)
    open(filepath, "w") do f
        println(f, "\\begin{table}[htbp]")
        println(f, "\\centering")
        println(f, "\\caption{Mean Reward Comparison by Solver and Problem Size}")
        println(f, "\\begin{tabular}{llrrrr}")
        println(f, "\\hline")
        println(f, "Problem Size & Solver & Mean & Std Dev & Min & Max \\\\")
        println(f, "\\hline")
        
        for row in eachrow(df)
            println(f, "$(row["Problem Size"]) & $(row["Solver"]) & " *
                      @sprintf("%.2f", row["Mean Reward"]) * " & " *
                      @sprintf("%.2f", row["Std Dev"]) * " & " *
                      @sprintf("%.2f", row["Min"]) * " & " *
                      @sprintf("%.2f", row["Max"]) * " \\\\")
        end
        
        println(f, "\\hline")
        println(f, "\\end{tabular}")
        println(f, "\\end{table}")
    end
    
    println("LaTeX reward table saved to: $filepath")
end

"""
Create LaTeX formatted statistics table.
"""
function create_latex_statistics_table(df::DataFrame, filepath::String)
    open(filepath, "w") do f
        println(f, "\\begin{table}[htbp]")
        println(f, "\\centering")
        println(f, "\\caption{Performance Statistics by Solver and Problem Size}")
        println(f, "\\begin{tabular}{llrrrrrr}")
        println(f, "\\hline")
        println(f, "Size & Solver & \\multicolumn{2}{c}{Announcement Changes} & Final Error & Incorrect & \\multicolumn{2}{c}{Change Magnitude} \\\\")
        println(f, " & & Mean & Std & Mean & (\\%) & Mean & Std \\\\")
        println(f, "\\hline")
        
        for row in eachrow(df)
            println(f, "$(row["Problem Size"]) & $(row["Solver"]) & " *
                      @sprintf("%.2f", row["Avg Announcement Changes"]) * " & " *
                      @sprintf("%.2f", row["Std Announcement Changes"]) * " & " *
                      @sprintf("%.2f", row["Avg Final Error"]) * " & " *
                      @sprintf("%.1f", row["Incorrect Final (%)"]) * " & " *
                      @sprintf("%.2f", row["Avg Change Magnitude"]) * " & " *
                      @sprintf("%.2f", row["Std Change Magnitude"]) * " \\\\")
        end
        
        println(f, "\\hline")
        println(f, "\\end{tabular}")
        println(f, "\\end{table}")
    end
    
    println("LaTeX statistics table saved to: $filepath")
end

"""
Create additional statistics plots
"""
function create_statistics_plots(df::DataFrame, problem_sizes, solvers, output_dir)
    stats_plot_dir = joinpath(output_dir, "statistics_plots")
    mkpath(stats_plot_dir)
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    sorted_solvers = sort_solvers(solvers)
    
    # Plot average number of changes
    p1 = Plots.plot(xlabel = "Problem Size", 
              ylabel = "Average Changes",
              size = (800, 600);
              PLOT_SETTINGS...)
    
    for solver in sorted_solvers
        formatted_solver = format_solver_name(solver)
        solver_data = filter(row -> row["Solver"] == formatted_solver, df)
        
        y_data = []
        x_data = []
        for size in sorted_sizes
            formatted_size = format_problem_size(size)
            if formatted_size in solver_data[!, "Problem Size"]
                size_row = solver_data[solver_data[!, "Problem Size"] .== formatted_size, :]
                if !isempty(size_row)
                    push!(y_data, size_row[1, "Avg Announcement Changes"])
                    push!(x_data, formatted_size)
                end
            end
        end
        
        if !isempty(y_data)
            plot!(p1, x_data, y_data,
                  label = format_solver_name(solver), 
                  marker = :circle, 
                  markersize = 6,
                  linewidth = 2,
                  color = get_solver_color(solver),
                  markerstrokewidth = 0)
        end
    end
    
    Plots.savefig(p1, joinpath(stats_plot_dir, "avg_announcement_changes.pdf"))
    
    # Plot average final error
    p2 = Plots.plot(xlabel = "Problem Size",
              ylabel = "Average Error", 
              size = (800, 600),
              legend = :right;
              PLOT_SETTINGS...)

    for solver in sorted_solvers
        formatted_solver = format_solver_name(solver)
        solver_data = filter(row -> row["Solver"] == formatted_solver, df)
        
        y_data = []
        x_data = []
        for size in sorted_sizes
            formatted_size = format_problem_size(size)
            if formatted_size in solver_data[!, "Problem Size"]
                size_row = solver_data[solver_data[!, "Problem Size"] .== formatted_size, :]
                if !isempty(size_row)
                    push!(y_data, size_row[1, "Avg Final Error"])
                    push!(x_data, formatted_size)
                end
            end
        end
        
        if !isempty(y_data)
            plot!(p2, x_data, y_data,
                  label = format_solver_name(solver),
                  marker = :circle,
                  markersize = 6, 
                  linewidth = 2,
                  color = get_solver_color(solver),
                  markerstrokewidth = 0)
        end
    end
    
    Plots.savefig(p2, joinpath(stats_plot_dir, "avg_final_error.pdf"))
    
    # Plot percentage of incorrect final predictions
    p3 = Plots.plot(xlabel = "Problem Size",
              ylabel = "Incorrect (%)",
              size = (800, 600);
              PLOT_SETTINGS...)
    
    for solver in sorted_solvers
        formatted_solver = format_solver_name(solver)
        solver_data = filter(row -> row["Solver"] == formatted_solver, df)
        
        y_data = []
        x_data = []
        for size in sorted_sizes
            formatted_size = format_problem_size(size)
            if formatted_size in solver_data[!, "Problem Size"]
                size_row = solver_data[solver_data[!, "Problem Size"] .== formatted_size, :]
                if !isempty(size_row)
                    push!(y_data, size_row[1, "Incorrect Final (%)"])
                    push!(x_data, formatted_size)
                end
            end
        end
        
        if !isempty(y_data)
            plot!(p3, x_data, y_data,
                  label = format_solver_name(solver),
                  marker = :circle,
                  markersize = 6,
                  linewidth = 2,
                  color = get_solver_color(solver),
                  markerstrokewidth = 0)
        end
    end
    
    Plots.savefig(p3, joinpath(stats_plot_dir, "incorrect_predictions.pdf"))
end

"""
Generate combined visualizations for the paper
"""
function generate_combined_plots(results, problem_sizes, solvers, output_dir)
    println("Generating combined visualizations...")
    
    combined_dir = joinpath(output_dir, "combined_plots")
    mkpath(combined_dir)
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    sorted_solvers = sort_solvers(solvers)
    
    # Create a 2x2 subplot of key metrics
    p1 = Plots.plot(legend = :bottomleft,
                    ylabel = "Mean Reward";
                    PLOT_SETTINGS...)
    p2 = Plots.plot(legend = :topright,
                    ylabel = "Error Rate (%)";
                    PLOT_SETTINGS...)  
    p3 = Plots.plot(legend = :topright,
                    ylabel = "Avg Changes";
                    PLOT_SETTINGS...)
    p4 = Plots.plot(legend = :topright, 
                    yscale = :log10,
                    ylabel = "Time (seconds)";
                    PLOT_SETTINGS...)
    
    for solver in sorted_solvers
        # Collect data across problem sizes
        mean_rewards = []
        error_rates = []
        avg_changes = []
        policy_times = []
        available_sizes = []
        
        for size in sorted_sizes
            if haskey(results, size) && haskey(results[size], solver)
                solver_results = results[size][solver]
                
                push!(mean_rewards, mean(solver_results["rewards"]))
                
                # Error rate = percentage with final error > 0
                error_rate = 100.0 * count(e -> e > 0, solver_results["final_errors"]) / 
                             length(solver_results["final_errors"])
                push!(error_rates, error_rate)
                
                push!(avg_changes, mean(solver_results["num_changes"]))
                push!(policy_times, get(solver_results, "policy_solve_time", 0.001))
                push!(available_sizes, format_problem_size(size))
            end
        end
        
        solver_color = get_solver_color(solver)
        formatted_solver_name = format_solver_name(solver)
        
        plot!(p1, available_sizes, mean_rewards, 
              label = formatted_solver_name, marker = :circle, markersize = 6, 
              linewidth = 2, color = solver_color, markerstrokewidth = 0)
        plot!(p2, available_sizes, error_rates, 
              label = formatted_solver_name, marker = :circle, markersize = 6, 
              linewidth = 2, color = solver_color, markerstrokewidth = 0)
        plot!(p3, available_sizes, avg_changes, 
              label = formatted_solver_name, marker = :circle, markersize = 6, 
              linewidth = 2, color = solver_color, markerstrokewidth = 0)
        plot!(p4, available_sizes, policy_times, 
              label = formatted_solver_name, marker = :circle, markersize = 6, 
              linewidth = 2, color = solver_color, markerstrokewidth = 0)
    end
    
    xlabel!(p1, "Problem Size")
    xlabel!(p2, "Problem Size")
    xlabel!(p3, "Problem Size") 
    xlabel!(p4, "Problem Size")
    
    combined = Plots.plot(p1, p2, p3, p4, layout = (2, 2), 
                         size = (1400, 1000);
                         PLOT_SETTINGS...)
    
    Plots.savefig(combined, joinpath(combined_dir, "key_metrics_comparison.pdf"))
end

"""
Reconstruct consolidated results from detailed batch files.
"""
function reconstruct_from_detailed_data(experiment_dir::String)
    detailed_dir = joinpath(experiment_dir, "detailed_data")
    if !isdir(detailed_dir)
        error("No results found in experiment directory: $experiment_dir")
    end
    
    all_results = Dict()
    
    # Iterate through problem sizes
    for size_name in readdir(detailed_dir)
        size_dir = joinpath(detailed_dir, size_name)
        if !isdir(size_dir)
            continue
        end
        
        size_results = Dict()
        
        # Iterate through solvers
        for solver_name in readdir(size_dir)
            solver_dir = joinpath(size_dir, solver_name)
            if !isdir(solver_dir)
                continue
            end
            
            # Load consolidated batch files
            consolidated_files = filter(f -> startswith(f, "consolidated_batch"), readdir(solver_dir))
            
            if isempty(consolidated_files)
                continue
            end
            
            # Aggregate metrics from all batches
            aggregated_metrics = Dict(
                "rewards" => Float64[],
                "initial_errors" => Int[],
                "final_errors" => Int[],
                "num_changes" => Int[],
                "avg_change_magnitudes" => Float64[],
                "std_change_magnitudes" => Float64[],
                "final_undershoot" => Bool[]
            )
            
            for batch_file in consolidated_files
                batch_path = joinpath(solver_dir, batch_file)
                try
                    batch_data = JSON.parsefile(batch_path)
                    if haskey(batch_data, "metrics")
                        metrics = batch_data["metrics"]
                        for (key, values) in aggregated_metrics
                            if haskey(metrics, key)
                                append!(values, metrics[key])
                            end
                        end
                    end
                catch e
                    println("Warning: Could not load batch file $batch_file: $e")
                end
            end
            
            size_results[solver_name] = aggregated_metrics
        end
        
        all_results[size_name] = size_results
    end
    
    return all_results
end

function generate_reward_analysis(results, problem_sizes, solvers, output_dir)
    println("Generating reward analysis...")
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    sorted_solvers = sort_solvers(solvers)
    
    # Prepare data for table
    table_data = []
    
    # Collect data by problem size
    for size in sorted_sizes
        for solver in sorted_solvers
            if haskey(results[size], solver)
                solver_data = results[size][solver]
                
                # Handle both old and new data formats
                rewards = if haskey(solver_data, "rewards")
                    solver_data["rewards"]
                elseif haskey(solver_data, "total_reward")  # Legacy format
                    [solver_data["total_reward"]]
                else
                    Float64[]  # No reward data
                end
                
                if !isempty(rewards)
                    push!(table_data, Dict(
                        "Problem Size" => format_problem_size(size),
                        "Solver" => format_solver_name(solver),
                        "Mean Reward" => mean(rewards),
                        "Std Dev" => length(rewards) > 1 ? std(rewards) : 0.0,
                        "Min" => minimum(rewards),
                        "Max" => maximum(rewards),
                        "Count" => length(rewards)
                    ))
                end
            end
        end
    end
    
    if isempty(table_data)
        println("Warning: No reward data found for analysis")
        return
    end
    
    # Convert to DataFrame for easier manipulation
    df = DataFrame(table_data)
    
    # Save as CSV
    CSV.write(joinpath(output_dir, "reward_statistics.csv"), df)
    
    # Create formatted LaTeX table
    create_latex_reward_table(df, joinpath(output_dir, "reward_table.tex"))
    
    # Create bar plot with error bars for each problem size
    for size in sorted_sizes
        formatted_size = format_problem_size(size)
        size_df = filter(row -> row["Problem Size"] == formatted_size, df)
        
        if !isempty(size_df)
            p = Plots.bar(
                size_df[!, "Solver"],
                size_df[!, "Mean Reward"],
                yerr = size_df[!, "Std Dev"],
                xlabel = "Solver",
                ylabel = "Mean Reward",
                legend = false,
                size = (800, 600),
                rotation = 45,
                fillalpha = 0.8,
                color = [get_solver_color(solver) for solver in size_df[!, "Solver"]];
                PLOT_SETTINGS...)
            
            Plots.savefig(p, joinpath(output_dir, "reward_comparison_$(size).pdf"))
        end
    end
    
    # Combined plot across all problem sizes - IMPROVED VERSION
    # Use GR backend for grouped bar charts
    current_backend = Plots.backend()
    gr()  
    
    # Use the predefined solver order directly, filtering for what exists in the data
    df_solvers = unique(df[!, "Solver"])
    unique_solvers = [format_solver_name(solver) for solver in SOLVER_ORDER if format_solver_name(solver) in df_solvers]
    formatted_sizes = [format_problem_size(size) for size in sorted_sizes]
    
    mean_matrix = zeros(length(unique_solvers), length(sorted_sizes))
    std_matrix = zeros(length(unique_solvers), length(sorted_sizes))
    
    for (i, solver) in enumerate(unique_solvers)
        for (j, size) in enumerate(formatted_sizes)
            solver_data = filter(row -> row["Solver"] == solver && row["Problem Size"] == size, df)
            if !isempty(solver_data)
                mean_matrix[i, j] = solver_data[1, "Mean Reward"]
                std_matrix[i, j] = solver_data[1, "Std Dev"]
            end
        end
    end
    
    # Create color palette with consistent colors
    solver_colors = [get_solver_color(solver) for solver in unique_solvers]
    
    p_combined = groupedbar(
        mean_matrix',
        bar_position = :dodge,
        bar_width = 0.7,
        yerr = std_matrix',
        labels = reshape(unique_solvers, 1, :),
        xticks = (1:length(formatted_sizes), formatted_sizes),
        xlabel = "Problem Size",
        ylabel = "Mean Reward",
        size = (1200, 700),
        legend = :bottomleft,
        color = reshape(solver_colors, 1, :),
        ylims = (-550, :auto);
        PLOT_SETTINGS...)
    
    # Add error bar legend entry
    plot!(p_combined, [NaN, NaN], [NaN, NaN], 
          label="±1σ", color=:black, linewidth=2, 
          linestyle=:solid, marker=:none)
    
    Plots.savefig(p_combined, joinpath(output_dir, "reward_comparison_combined.pdf"))
    
    # Restore previous backend
    if current_backend != Plots.GRBackend()
        eval(current_backend.backend_name)()
    end
    
    println("Note: Error bars in reward comparison plots indicate ± standard deviation")
end

"""
Enhanced histogram generation with improved formatting.
"""
function generate_reward_histograms(results, problem_sizes, solvers, output_dir)
    println("Generating reward histograms...")
    
    hist_dir = joinpath(output_dir, "histograms")
    mkpath(hist_dir)
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    sorted_solvers = sort_solvers(solvers)

    # Individual histograms for each problem size and solver
    for size in sorted_sizes
        for solver in sorted_solvers
            if haskey(results[size], solver)
                solver_data = results[size][solver]
                
                # Handle both old and new data formats
                rewards = if haskey(solver_data, "rewards")
                    solver_data["rewards"]
                elseif haskey(solver_data, "total_reward")
                    [solver_data["total_reward"]]
                else
                    continue  # Skip if no reward data
                end
                
                if length(rewards) > 1  # Only create histogram if we have multiple data points
                    p = Plots.histogram(
                        rewards,
                        bins = min(20, length(rewards)),
                        xlabel = "Total Reward",
                        ylabel = "Frequency",
                        legend = false,
                        fillalpha = 0.7,
                        color = get_solver_color(solver),
                        size = (800, 600);
                        PLOT_SETTINGS...
                    )
                    
                    # Add mean and median lines
                    vline!([mean(rewards)], label = "Mean", linewidth = 2, color = :red)
                    if length(rewards) > 1
                        vline!([median(rewards)], label = "Median", linewidth = 2, color = :green, linestyle = :dash)
                    end
                    
                    Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_$(solver).pdf"))
                end
            end
        end
    end
    
    # Combined histograms by problem size
    for size in sorted_sizes
        p = Plots.plot(
            xlabel = "Total Reward",
            ylabel = "Frequency",
            size = (1000, 600);
            PLOT_SETTINGS...
        )
        
        plot_created = false
        for solver in solvers
            if haskey(results[size], solver)
                solver_data = results[size][solver]
                rewards = get(solver_data, "rewards", [])
                
                if length(rewards) > 1
                    Plots.histogram!(
                        p,
                        rewards,
                        bins = min(20, length(rewards)),
                        alpha = 0.5,
                        label = format_solver_name(solver),
                        color = get_solver_color(solver)
                    )
                    plot_created = true
                end
            end
        end
        
        if plot_created
            Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_combined.pdf"))
        end
    end
end

"""
Generate comprehensive statistics table
"""
function generate_statistics_table(results, problem_sizes, solvers, output_dir)
    println("Generating statistics table...")
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    sorted_solvers = sort_solvers(solvers)
    
    # Collect all statistics
    stats_data = []
    
    for size in sorted_sizes
        for solver in sorted_solvers
            if haskey(results[size], solver)
                solver_results = results[size][solver]
                
                # Extract metrics
                rewards = solver_results["rewards"]
                num_changes = solver_results["num_changes"]
                final_errors = solver_results["final_errors"]
                avg_change_mags = solver_results["avg_change_magnitudes"]
                
                # Count incorrect final predictions (final error > 0)
                incorrect_final = count(e -> e > 0, final_errors)
                
                # Compute statistics
                stats = Dict(
                    "Problem Size" => format_problem_size(size),
                    "Solver" => format_solver_name(solver),
                    "Avg Announcement Changes" => mean(num_changes),
                    "Std Announcement Changes" => std(num_changes),
                    "Avg Final Error" => mean(final_errors),
                    "Incorrect Final (%)" => 100.0 * incorrect_final / length(final_errors),
                    "Avg Change Magnitude" => mean(avg_change_mags),
                    "Std Change Magnitude" => std(avg_change_mags),
                    "Policy Time (s)" => get(solver_results, "policy_solve_time", 0.0)
                )
                
                push!(stats_data, stats)
            end
        end
    end
    
    # Convert to DataFrame
    df = DataFrame(stats_data)
    
    # Save as CSV
    CSV.write(joinpath(output_dir, "comparison_statistics.csv"), df)
    
    # Create LaTeX table
    create_latex_statistics_table(df, joinpath(output_dir, "statistics_table.tex"))
    
    # Create summary plots
    create_statistics_plots(df, sorted_sizes, solvers, output_dir)
end

"""
Generate memory usage report from the experiment.
"""
function generate_memory_report(experiment_dir::String, output_dir::String)
    println("Generating memory usage report...")
    
    report = []
    total_files = 0
    total_size_mb = 0.0
    
    # Analyze file structure
    if isdir(experiment_dir)
        for (root, dirs, files) in walkdir(experiment_dir)
            for file in files
                filepath = joinpath(root, file)
                if isfile(filepath)
                    total_files += 1
                    file_size_mb = stat(filepath).size / (1024 * 1024)
                    total_size_mb += file_size_mb
                    
                    # Categorize files
                    rel_path = relpath(filepath, experiment_dir)
                    category = if endswith(file, ".json") && contains(rel_path, "detailed_data")
                        "Detailed Data"
                    elseif endswith(file, ".json") && !contains(rel_path, "detailed_data")
                        "Consolidated Results"
                    elseif endswith(file, ".png") || endswith(file, ".svg") || endswith(file, ".pdf")
                        "Plots"
                    elseif endswith(file, ".jld2")
                        "Policies"
                    else
                        "Other"
                    end
                    
                    push!(report, Dict(
                        "file" => rel_path,
                        "size_mb" => file_size_mb,
                        "category" => category
                    ))
                end
            end
        end
    end
    
    # Create summary
    summary = Dict(
        "total_files" => total_files,
        "total_size_mb" => total_size_mb,
        "files" => report
    )
    
    # Save detailed report
    report_path = joinpath(output_dir, "memory_usage_report.json")
    open(report_path, "w") do f
        JSON.print(f, summary, 4)
    end
    
    # Create summary visualization
    if !isempty(report)
        df = DataFrame(report)
        
        # Group by category
        category_summary = combine(groupby(df, :category), 
                                 :size_mb => sum => :total_size_mb,
                                 nrow => :file_count)
        
        # Create pie chart of disk usage by category
        p = Plots.pie(
            category_summary.category,
            category_summary.total_size_mb,
            legend = :outertopleft,
            size = (800, 600);
            PLOT_SETTINGS...
        )
        
        Plots.savefig(p, joinpath(output_dir, "disk_usage_by_category.pdf"))
        
        # Save category summary
        CSV.write(joinpath(output_dir, "disk_usage_summary.csv"), category_summary)
        
        println("Memory usage report saved to: $report_path")
        println("Total disk usage: $(round(total_size_mb, digits=1)) MB across $total_files files")
    end
end

"""
Main analysis function
"""
function analyze_results(experiment_dir::String; output_dir::Union{String, Nothing}=nothing)
    # Use experiment directory for output if not specified
    if output_dir === nothing
        output_dir = joinpath(experiment_dir, "analysis")
    end
    mkpath(output_dir)
    
    # Set default backend to GR for consistency
    gr()
    
    # Load experiment configuration
    config_path = joinpath(experiment_dir, "experiment_config.json")
    if !isfile(config_path)
        error("Experiment configuration not found: $config_path")
    end
    config = JSON.parsefile(config_path)
    
    # Load all results (handles both old and new formats)
    all_results = load_incremental_results(experiment_dir)
    
    if isempty(all_results)
        error("No results found in experiment directory: $experiment_dir")
    end
    
    problem_sizes = collect(keys(all_results))
    
    # Extract solvers from config or infer from data
    if haskey(config, "solvers")
        solvers = config["solvers"]
    else
        # Infer solvers from first problem size
        first_size = first(values(all_results))
        solvers = collect(keys(first_size))
    end

    # Sort for consistent presentation
    sorted_sizes = sort_problem_sizes(problem_sizes)
    sorted_solvers = sort_solvers(solvers)
    
    println("Analyzing results for:")
    println("  Problem sizes: $(join(sorted_sizes, ", "))")
    println("  Solvers: $(join(sorted_solvers, ", "))")
    println("  Data structure: $(haskey(config, "save_frequency") ? "Incremental" : "Legacy")")
    println("  Output format: SVG and PDF (high quality)")
    println()
    
    # 1. Generate reward comparison table and plots
    generate_reward_analysis(all_results, problem_sizes, solvers, output_dir)
    
    # 2. Generate histogram distributions
    generate_reward_histograms(all_results, problem_sizes, solvers, output_dir)
    
    # 3. Generate comparison statistics table
    generate_statistics_table(all_results, problem_sizes, solvers, output_dir)
    
    # 4. Generate combined visualizations
    generate_combined_plots(all_results, problem_sizes, solvers, output_dir)
    
    # 5. Generate memory usage report if available
    generate_memory_report(experiment_dir, output_dir)
    
    # 6. Generate belief evolution plots from experiment JSON data (new feature)
    if REGEN_BELIEF_HISTORIES == true
        generate_belief_evolution_plots_from_json(experiment_dir, output_dir)
    end
    
    println("\nAnalysis complete! Results saved to: $output_dir")
end

# Main execution with enhanced error handling
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia analyze_paper_results.jl <experiment_directory> [output_directory]")
        println("       Supports both legacy and incremental data structures")
        exit(1)
    end
    
    experiment_dir = ARGS[1]
    output_dir = length(ARGS) >= 2 ? ARGS[2] : nothing
    
    if !isdir(experiment_dir)
        println("Error: Experiment directory not found: $experiment_dir")
        exit(1)
    end
    
    try
        analyze_results(experiment_dir, output_dir=output_dir)
        println("✓ Analysis completed successfully")
    catch e
        println("✗ Error during analysis: $e")
        if isa(e, InterruptException)
            println("Analysis interrupted by user")
        else
            println("Stack trace:")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
        exit(1)
    end
end