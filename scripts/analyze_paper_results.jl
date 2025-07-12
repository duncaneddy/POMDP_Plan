#!/usr/bin/env julia

# Generate all plots and tables for the paper

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

"""
Main analysis function that generates all plots and tables from experiment results.
"""
function analyze_results(experiment_dir::String; output_dir::Union{String, Nothing}=nothing)
    # Use experiment directory for output if not specified
    if output_dir === nothing
        output_dir = joinpath(experiment_dir, "analysis")
    end
    mkpath(output_dir)
    
    # Load experiment configuration
    config = JSON.parsefile(joinpath(experiment_dir, "experiment_config.json"))
    
    # Load all results
    all_results = JSON.parsefile(joinpath(experiment_dir, "all_results.json"))
    
    problem_sizes = collect(keys(all_results))
    solvers = config["solvers"]
    
    println("Analyzing results for:")
    println("  Problem sizes: $(join(problem_sizes, ", "))")
    println("  Solvers: $(join(solvers, ", "))")
    println()
    
    # 1. Generate reward comparison table and plots
    generate_reward_analysis(all_results, problem_sizes, solvers, output_dir)
    
    # 2. Generate histogram distributions
    generate_reward_histograms(all_results, problem_sizes, solvers, output_dir)
    
    # 3. Generate comparison statistics table
    generate_statistics_table(all_results, problem_sizes, solvers, output_dir)
    
    # 4. Generate combined visualizations
    generate_combined_plots(all_results, problem_sizes, solvers, output_dir)
    
    println("\nAnalysis complete! Results saved to: $output_dir")
end

"""
Generate reward comparison table and plots showing mean and std deviation.
"""
function generate_reward_analysis(results, problem_sizes, solvers, output_dir)
    println("Generating reward analysis...")
    
    # Prepare data for table
    table_data = []
    
    # Collect data by problem size
    for size in problem_sizes
        for solver in solvers
            rewards = results[size][solver]["rewards"]
            push!(table_data, Dict(
                "Problem Size" => size,
                "Solver" => solver,
                "Mean Reward" => mean(rewards),
                "Std Dev" => std(rewards),
                "Min" => minimum(rewards),
                "Max" => maximum(rewards),
                "Count" => length(rewards)
            ))
        end
    end
    
    # Convert to DataFrame for easier manipulation
    df = DataFrame(table_data)
    
    # Save as CSV
    CSV.write(joinpath(output_dir, "reward_statistics.csv"), df)
    
    # Create formatted LaTeX table
    create_latex_reward_table(df, joinpath(output_dir, "reward_table.tex"))
    
    # Create bar plot with error bars for each problem size
    for size in problem_sizes
        size_df = filter(row -> row["Problem Size"] == size, df)
        
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
            bottom_margin = 10Plots.mm,
            fillalpha = 0.8
        )
        
        Plots.savefig(p, joinpath(output_dir, "reward_comparison_$(size).png"))
    end
    
    # Combined plot across all problem sizes
    gr()  # Use GR backend for grouped bar charts
    
    # Reshape data for grouped bar chart
    mean_matrix = zeros(length(solvers), length(problem_sizes))
    std_matrix = zeros(length(solvers), length(problem_sizes))
    
    for (i, solver) in enumerate(solvers)
        for (j, size) in enumerate(problem_sizes)
            solver_data = filter(row -> row["Solver"] == solver && row["Problem Size"] == size, df)
            if !isempty(solver_data)
                mean_matrix[i, j] = solver_data[1, "Mean Reward"]
                std_matrix[i, j] = solver_data[1, "Std Dev"]
            end
        end
    end
    
    p_combined = groupedbar(
        mean_matrix',
        bar_position = :dodge,
        bar_width = 0.7,
        yerr = std_matrix',
        labels = reshape(solvers, 1, :),
        xticks = (1:length(problem_sizes), problem_sizes),
        title = "Mean Reward Comparison Across Problem Sizes",
        xlabel = "Problem Size",
        ylabel = "Mean Reward",
        size = (1000, 600),
        legend = :bottomleft
    )
    
    Plots.savefig(p_combined, joinpath(output_dir, "reward_comparison_combined.png"))
end

"""
Generate histogram distributions of rewards.
"""
function generate_reward_histograms(results, problem_sizes, solvers, output_dir)
    println("Generating reward histograms...")
    
    hist_dir = joinpath(output_dir, "histograms")
    mkpath(hist_dir)
    
    # Individual histograms for each problem size and solver
    for size in problem_sizes
        for solver in solvers
            rewards = results[size][solver]["rewards"]
            
            p = Plots.histogram(
                rewards,
                bins = 20,
                title = "Reward Distribution - $solver ($size)",
                xlabel = "Total Reward",
                ylabel = "Frequency",
                legend = false,
                fillalpha = 0.7,
                color = :blue
            )
            
            # Add mean and median lines
            vline!([mean(rewards)], label = "Mean", linewidth = 2, color = :red)
            vline!([median(rewards)], label = "Median", linewidth = 2, color = :green, linestyle = :dash)
            
            Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_$(solver).png"))
        end
    end
    
    # Combined histograms by problem size
    for size in problem_sizes
        p = Plots.plot(
            title = "Reward Distributions - $size Problem",
            xlabel = "Total Reward",
            ylabel = "Frequency",
            size = (1000, 600)
        )
        
        for (i, solver) in enumerate(solvers)
            rewards = results[size][solver]["rewards"]
            Plots.histogram!(
                p,
                rewards,
                bins = 20,
                alpha = 0.5,
                label = solver,
                color = i
            )
        end
        
        Plots.savefig(p, joinpath(hist_dir, "hist_$(size)_combined.png"))
    end
    
    # Combined histogram across all problem sizes for each solver
    for solver in solvers
        p = Plots.plot(
            title = "Reward Distributions - $solver",
            xlabel = "Total Reward", 
            ylabel = "Frequency",
            size = (1000, 600)
        )
        
        for (i, size) in enumerate(problem_sizes)
            rewards = results[size][solver]["rewards"]
            Plots.histogram!(
                p,
                rewards,
                bins = 20,
                alpha = 0.5,
                label = size,
                color = i
            )
        end
        
        Plots.savefig(p, joinpath(hist_dir, "hist_$(solver)_all_sizes.png"))
    end
end

"""
Generate comprehensive statistics table.
"""
function generate_statistics_table(results, problem_sizes, solvers, output_dir)
    println("Generating statistics table...")
    
    # Collect all statistics
    stats_data = []
    
    for size in problem_sizes
        for solver in solvers
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
    
    # Convert to DataFrame
    df = DataFrame(stats_data)
    
    # Save as CSV
    CSV.write(joinpath(output_dir, "comparison_statistics.csv"), df)
    
    # Create LaTeX table
    create_latex_statistics_table(df, joinpath(output_dir, "statistics_table.tex"))
    
    # Create summary plots
    create_statistics_plots(df, problem_sizes, solvers, output_dir)
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
    
    # Plot average number of changes
    p1 = Plots.plot(title = "Average Number of Announcement Changes",
              xlabel = "Problem Size", 
              ylabel = "Average Changes",
              size = (800, 600))
    
    for solver in solvers
        solver_data = filter(row -> row["Solver"] == solver, df)
        plot!(p1, problem_sizes, 
              [solver_data[solver_data[!, "Problem Size"] .== size, "Avg Announcement Changes"][1] 
               for size in problem_sizes],
              label = solver, 
              marker = :circle, 
              markersize = 6,
              linewidth = 2)
    end
    
    Plots.savefig(p1, joinpath(stats_plot_dir, "avg_announcement_changes.png"))
    
    # Plot average final error
    p2 = Plots.plot(title = "Average Final Error",
              xlabel = "Problem Size",
              ylabel = "Average Error", 
              size = (800, 600))
    
    for solver in solvers
        solver_data = filter(row -> row["Solver"] == solver, df)
        plot!(p2, problem_sizes,
              [solver_data[solver_data[!, "Problem Size"] .== size, "Avg Final Error"][1]
               for size in problem_sizes],
              label = solver,
              marker = :circle,
              markersize = 6, 
              linewidth = 2)
    end
    
    Plots.savefig(p2, joinpath(stats_plot_dir, "avg_final_error.png"))
    
    # Plot percentage of incorrect final predictions
    p3 = Plots.plot(title = "Percentage of Incorrect Final Predictions",
              xlabel = "Problem Size",
              ylabel = "Incorrect (%)",
              size = (800, 600))
    
    for solver in solvers
        solver_data = filter(row -> row["Solver"] == solver, df)
        plot!(p3, problem_sizes,
              [solver_data[solver_data[!, "Problem Size"] .== size, "Incorrect Final (%)"][1]
               for size in problem_sizes],
              label = solver,
              marker = :circle,
              markersize = 6,
              linewidth = 2)
    end
    
    Plots.savefig(p3, joinpath(stats_plot_dir, "incorrect_predictions.png"))
end

"""
Generate combined visualizations for the paper.
"""
function generate_combined_plots(results, problem_sizes, solvers, output_dir)
    println("Generating combined visualizations...")
    
    combined_dir = joinpath(output_dir, "combined_plots")
    mkpath(combined_dir)
    
    # Create a 2x2 subplot of key metrics
    p1 = Plots.plot(title = "Mean Reward", legend = :bottomleft)
    p2 = Plots.plot(title = "Final Error Rate", legend = :topright)  
    p3 = Plots.plot(title = "Announcement Changes", legend = :topright)
    p4 = Plots.plot(title = "Policy Generation Time", legend = :topright, yscale = :log10)
    
    for solver in solvers
        # Collect data across problem sizes
        mean_rewards = []
        error_rates = []
        avg_changes = []
        policy_times = []
        
        for size in problem_sizes
            solver_results = results[size][solver]
            
            push!(mean_rewards, mean(solver_results["rewards"]))
            
            # Error rate = percentage with final error > 0
            error_rate = 100.0 * count(e -> e > 0, solver_results["final_errors"]) / 
                         length(solver_results["final_errors"])
            push!(error_rates, error_rate)
            
            push!(avg_changes, mean(solver_results["num_changes"]))
            push!(policy_times, get(solver_results, "policy_solve_time", 0.001))
        end
        
        plot!(p1, problem_sizes, mean_rewards, label = solver, marker = :circle, markersize = 6)
        plot!(p2, problem_sizes, error_rates, label = solver, marker = :circle, markersize = 6)
        plot!(p3, problem_sizes, avg_changes, label = solver, marker = :circle, markersize = 6)
        plot!(p4, problem_sizes, policy_times, label = solver, marker = :circle, markersize = 6)
    end
    
    xlabel!(p1, "Problem Size")
    ylabel!(p1, "Mean Reward")
    
    xlabel!(p2, "Problem Size")
    ylabel!(p2, "Error Rate (%)")
    
    xlabel!(p3, "Problem Size") 
    ylabel!(p3, "Avg Changes")
    
    xlabel!(p4, "Problem Size")
    ylabel!(p4, "Time (seconds)")
    
    combined = Plots.plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
    Plots.savefig(combined, joinpath(combined_dir, "key_metrics_comparison.png"))
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia analyze_paper_results.jl <experiment_directory> [output_directory]")
        exit(1)
    end
    
    experiment_dir = ARGS[1]
    output_dir = length(ARGS) >= 2 ? ARGS[2] : nothing
    
    analyze_results(experiment_dir, output_dir=output_dir)
end