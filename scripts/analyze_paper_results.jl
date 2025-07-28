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

# Define consistent problem size ordering and color scheme
const PROBLEM_SIZE_ORDER = ["small", "medium", "large", "xlarge"]
const SOLVER_COLORS = Dict(
    "OBSERVEDTIME" => :blue,
    "MOSTLIKELY" => :red, 
    "QMDP" => :green,
    "MOMDP_SARSOP" => :purple,
    "SARSOP" => :orange,
    "POMCPOW" => :brown,
    "FIB" => :pink,
    "PBVI" => :gray
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
    :dpi => 300
)

"""
Get consistent color for a solver across all plots.
"""
function get_solver_color(solver::String)
    return get(SOLVER_COLORS, solver, :black)
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
Create additional statistics plots.
"""
function create_statistics_plots(df::DataFrame, problem_sizes, solvers, output_dir)
    stats_plot_dir = joinpath(output_dir, "statistics_plots")
    mkpath(stats_plot_dir)
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    
    # Plot average number of changes
    p1 = Plots.plot(title = "Average Number of Announcement Changes",
              xlabel = "Problem Size", 
              ylabel = "Average Changes",
              size = (800, 600);
              PLOT_SETTINGS...)
    
    for solver in solvers
        solver_data = filter(row -> row["Solver"] == solver, df)
        if !isempty(solver_data)
            y_data = []
            x_data = []
            for size in sorted_sizes
                size_data = solver_data[solver_data[!, "Problem Size"] .== size, :]
                if !isempty(size_data)
                    push!(y_data, size_data[1, "Avg Announcement Changes"])
                    push!(x_data, size)
                end
            end
            
            if !isempty(x_data)
                plot!(p1, x_data, y_data,
                      label = solver, 
                      marker = :circle, 
                      markersize = 6,
                      linewidth = 2,
                      color = get_solver_color(solver))
            end
        end
    end
    
    Plots.savefig(p1, joinpath(stats_plot_dir, "avg_announcement_changes.svg"))
    Plots.savefig(p1, joinpath(stats_plot_dir, "avg_announcement_changes.pdf"))
    
    # Plot average final error
    p2 = Plots.plot(title = "Average Final Error",
              xlabel = "Problem Size",
              ylabel = "Average Error", 
              size = (800, 600);
              PLOT_SETTINGS...)
    
    for solver in solvers
        solver_data = filter(row -> row["Solver"] == solver, df)
        if !isempty(solver_data)
            y_data = []
            x_data = []
            for size in sorted_sizes
                size_data = solver_data[solver_data[!, "Problem Size"] .== size, :]
                if !isempty(size_data)
                    push!(y_data, size_data[1, "Avg Final Error"])
                    push!(x_data, size)
                end
            end
            
            if !isempty(x_data)
                plot!(p2, x_data, y_data,
                      label = solver,
                      marker = :circle,
                      markersize = 6, 
                      linewidth = 2,
                      color = get_solver_color(solver))
            end
        end
    end
    
    Plots.savefig(p2, joinpath(stats_plot_dir, "avg_final_error.svg"))
    Plots.savefig(p2, joinpath(stats_plot_dir, "avg_final_error.pdf"))
    
    # Plot percentage of incorrect final predictions
    p3 = Plots.plot(title = "Percentage of Incorrect Final Predictions",
              xlabel = "Problem Size",
              ylabel = "Incorrect (%)",
              size = (800, 600);
              PLOT_SETTINGS...)
    
    for solver in solvers
        solver_data = filter(row -> row["Solver"] == solver, df)
        if !isempty(solver_data)
            y_data = []
            x_data = []
            for size in sorted_sizes
                size_data = solver_data[solver_data[!, "Problem Size"] .== size, :]
                if !isempty(size_data)
                    push!(y_data, size_data[1, "Incorrect Final (%)"])
                    push!(x_data, size)
                end
            end
            
            if !isempty(x_data)
                plot!(p3, x_data, y_data,
                      label = solver,
                      marker = :circle,
                      markersize = 6,
                      linewidth = 2,
                      color = get_solver_color(solver))
            end
        end
    end
    
    Plots.savefig(p3, joinpath(stats_plot_dir, "incorrect_predictions.svg"))
    Plots.savefig(p3, joinpath(stats_plot_dir, "incorrect_predictions.pdf"))
end

"""
Generate combined visualizations for the paper with improved formatting.
"""
function generate_combined_plots(results, problem_sizes, solvers, output_dir)
    println("Generating combined visualizations...")
    
    combined_dir = joinpath(output_dir, "combined_plots")
    mkpath(combined_dir)
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    
    # Create a 2x2 subplot of key metrics
    p1 = Plots.plot(title = "Mean Reward", 
                    legend = :bottomleft,
                    ylabel = "Mean Reward";
                    PLOT_SETTINGS...)
    p2 = Plots.plot(title = "Final Error Rate", 
                    legend = :topright,
                    ylabel = "Error Rate (%)";
                    PLOT_SETTINGS...)  
    p3 = Plots.plot(title = "Announcement Changes", 
                    legend = :topright,
                    ylabel = "Avg Changes";
                    PLOT_SETTINGS...)
    p4 = Plots.plot(title = "Policy Generation Time", 
                    legend = :topright, 
                    yscale = :log10,
                    ylabel = "Time (seconds)";
                    PLOT_SETTINGS...)
    
    for solver in solvers
        # Collect data across problem sizes - only for sizes where data exists
        mean_rewards = Float64[]
        error_rates = Float64[]
        avg_changes = Float64[]
        policy_times = Float64[]
        available_sizes = String[]
        
        for size in sorted_sizes
            if haskey(results, size) && haskey(results[size], solver)
                solver_results = results[size][solver]
                
                # Check if the solver has meaningful data
                if haskey(solver_results, "rewards") && !isempty(solver_results["rewards"])
                    push!(mean_rewards, mean(solver_results["rewards"]))
                    
                    # Error rate = percentage with final error > 0
                    if haskey(solver_results, "final_errors") && !isempty(solver_results["final_errors"])
                        error_rate = 100.0 * count(e -> e > 0, solver_results["final_errors"]) / 
                                     length(solver_results["final_errors"])
                        push!(error_rates, error_rate)
                    else
                        push!(error_rates, NaN)
                    end
                    
                    if haskey(solver_results, "num_changes") && !isempty(solver_results["num_changes"])
                        push!(avg_changes, mean(solver_results["num_changes"]))
                    else
                        push!(avg_changes, NaN)
                    end
                    
                    push!(policy_times, max(get(solver_results, "policy_solve_time", 0.001), 0.001))
                    push!(available_sizes, size)
                end
            end
        end
        
        # Only plot if we have data
        if !isempty(available_sizes)
            solver_color = get_solver_color(solver)
            
            # Filter out NaN values for each plot separately
            if !all(isnan.(mean_rewards))
                valid_idx = .!isnan.(mean_rewards)
                if any(valid_idx)
                    plot!(p1, available_sizes[valid_idx], mean_rewards[valid_idx], 
                          label = solver, marker = :circle, markersize = 6, 
                          linewidth = 2, color = solver_color)
                end
            end
            
            if !all(isnan.(error_rates))
                valid_idx = .!isnan.(error_rates)
                if any(valid_idx)
                    plot!(p2, available_sizes[valid_idx], error_rates[valid_idx], 
                          label = solver, marker = :circle, markersize = 6, 
                          linewidth = 2, color = solver_color)
                end
            end
            
            if !all(isnan.(avg_changes))
                valid_idx = .!isnan.(avg_changes)
                if any(valid_idx)
                    plot!(p3, available_sizes[valid_idx], avg_changes[valid_idx], 
                          label = solver, marker = :circle, markersize = 6, 
                          linewidth = 2, color = solver_color)
                end
            end
            
            if !all(isnan.(policy_times)) && !all(policy_times .<= 0.001)
                valid_idx = .!isnan.(policy_times) .& (policy_times .> 0.001)
                if any(valid_idx)
                    plot!(p4, available_sizes[valid_idx], policy_times[valid_idx], 
                          label = solver, marker = :circle, markersize = 6, 
                          linewidth = 2, color = solver_color)
                end
            end
        end
    end
    
    xlabel!(p1, "Problem Size")
    xlabel!(p2, "Problem Size")
    xlabel!(p3, "Problem Size") 
    xlabel!(p4, "Problem Size")
    
    combined = Plots.plot(p1, p2, p3, p4, layout = (2, 2), 
                         size = (1400, 1000);
                         PLOT_SETTINGS...)
    
    Plots.savefig(combined, joinpath(combined_dir, "key_metrics_comparison.svg"))
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

"""
Enhanced reward analysis with improved formatting and consistent colors.
"""
function generate_reward_analysis(results, problem_sizes, solvers, output_dir)
    println("Generating reward analysis...")
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    
    # Prepare data for table
    table_data = []
    
    # Collect data by problem size
    for size in sorted_sizes
        for solver in solvers
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
                        "Problem Size" => size,
                        "Solver" => solver,
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
        size_df = filter(row -> row["Problem Size"] == size, df)
        
        if !isempty(size_df)
            # Sort by solver name for consistency
            sort!(size_df, :Solver)
            
            p = Plots.bar(
                size_df[!, "Solver"],
                size_df[!, "Mean Reward"],
                yerr = size_df[!, "Std Dev"],
                title = "Mean Reward by Solver - $size Problem",
                xlabel = "Solver",
                ylabel = "Mean Reward",
                legend = false,
                size = (800, 600),
                rotation = 45,
                fillalpha = 0.8,
                color = [get_solver_color(solver) for solver in size_df[!, "Solver"]];
                PLOT_SETTINGS...)
            
            Plots.savefig(p, joinpath(output_dir, "reward_comparison_$(size).svg"))
            Plots.savefig(p, joinpath(output_dir, "reward_comparison_$(size).pdf"))
        end
    end
    
    # Combined plot across all problem sizes - IMPROVED VERSION
    gr()  # Use GR backend for grouped bar charts
    
    # Reshape data for grouped bar chart with consistent ordering
    unique_solvers = sort(unique(df[!, "Solver"]))
    
    # Only include sizes that have data for at least one solver
    available_sizes = [size for size in sorted_sizes if any(row -> row["Problem Size"] == size, eachrow(df))]
    
    mean_matrix = zeros(length(unique_solvers), length(available_sizes))
    std_matrix = zeros(length(unique_solvers), length(available_sizes))
    
    for (i, solver) in enumerate(unique_solvers)
        for (j, size) in enumerate(available_sizes)
            solver_data = filter(row -> row["Solver"] == solver && row["Problem Size"] == size, df)
            if !isempty(solver_data)
                mean_matrix[i, j] = solver_data[1, "Mean Reward"]
                std_matrix[i, j] = solver_data[1, "Std Dev"]
            else
                # Use NaN for missing data - Plots.jl will handle this correctly
                mean_matrix[i, j] = NaN
                std_matrix[i, j] = NaN
            end
        end
    end
    
    # Filter out solvers that have no data at all
    has_data = [!all(isnan.(mean_matrix[i, :])) for i in 1:length(unique_solvers)]
    filtered_solvers = unique_solvers[has_data]
    filtered_mean_matrix = mean_matrix[has_data, :]
    filtered_std_matrix = std_matrix[has_data, :]
    
    if !isempty(filtered_solvers) && !isempty(available_sizes)
        p_combined = groupedbar(
            filtered_mean_matrix',
            bar_position = :dodge,
            bar_width = 0.7,
            yerr = filtered_std_matrix',  # YES, these lines indicate +/- standard deviation
            labels = reshape(filtered_solvers, 1, :),
            xticks = (1:length(available_sizes), available_sizes),
            title = "Mean Reward Comparison Across Problem Sizes",
            xlabel = "Problem Size",
            ylabel = "Mean Reward",
            size = (1200, 700),
            legend = :bottomleft;
            PLOT_SETTINGS...)
        
        Plots.savefig(p_combined, joinpath(output_dir, "reward_comparison_combined.svg"))
        Plots.savefig(p_combined, joinpath(output_dir, "reward_comparison_combined.pdf"))
    end
    
    println("Note: Error bars in reward comparison plots indicate ± standard deviation")
    println("Analysis complete!")
    println()
end

"""
Enhanced histogram generation.
"""
function generate_reward_histograms(results, problem_sizes, solvers, output_dir)
    println("Generating reward histograms...")
    
    hist_dir = joinpath(output_dir, "histograms")
    mkpath(hist_dir)
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    
    # Individual histograms for each problem size and solver
    for size in sorted_sizes
        if haskey(results, size)
            for solver in solvers
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
                            title = "Reward Distribution - $solver ($size)",
                            xlabel = "Total Reward",
                            ylabel = "Frequency",
                            legend = true,
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
                        
                        Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_$(solver).svg"))
                        Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_$(solver).pdf"))
                    end
                end
            end
        end
    end
    
    # Combined histograms by problem size
    for size in sorted_sizes
        if haskey(results, size)
            p = Plots.plot(
                title = "Reward Distributions - $size Problem",
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
                            label = solver,
                            color = get_solver_color(solver)
                        )
                        plot_created = true
                    end
                end
            end
            
            if plot_created
                Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_combined.svg"))
                Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_combined.pdf"))
            end
        end
    end
end

"""
Generate comprehensive statistics table.
"""
function generate_statistics_table(results, problem_sizes, solvers, output_dir)
    println("Generating statistics table...")
    
    # Sort problem sizes consistently
    sorted_sizes = sort_problem_sizes(problem_sizes)
    
    # Collect all statistics
    stats_data = []
    
    for size in sorted_sizes
        for solver in solvers
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
                    "Problem Size" => size,
                    "Solver" => solver,
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
            title = "Disk Usage by File Category",
            legend = :outertopleft,
            size = (800, 600);
            PLOT_SETTINGS...
        )
        
        Plots.savefig(p, joinpath(output_dir, "disk_usage_by_category.svg"))
        Plots.savefig(p, joinpath(output_dir, "disk_usage_by_category.pdf"))
        
        # Save category summary
        CSV.write(joinpath(output_dir, "disk_usage_summary.csv"), category_summary)
        
        println("Memory usage report saved to: $report_path")
        println("Total disk usage: $(round(total_size_mb, digits=1)) MB across $total_files files")
    end
end

"""
Main analysis function.
"""
function analyze_results(experiment_dir::String; output_dir::Union{String, Nothing}=nothing)
    # Use experiment directory for output if not specified
    if output_dir === nothing
        output_dir = joinpath(experiment_dir, "analysis")
    end
    mkpath(output_dir)
    
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
    
    println("Analyzing results for:")
    println("  Problem sizes: $(join(sort_problem_sizes(problem_sizes), ", "))")
    println("  Solvers: $(join(solvers, ", "))")
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
        println("✓ Analysis completed successfully with improved formatting and fixed warnings")
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