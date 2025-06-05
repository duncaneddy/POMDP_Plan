# Define the experiment running functionality
function run_experiment(
    min_end_time::Int,
    max_end_time::Int,
    solvers::Vector{String},
    num_simulations::Int,
    output_dir::String;
    fixed_true_end_time::Union{Int, Nothing}=nothing,
    initial_announce::Union{Int, Nothing}=0,
    discount_factor::Float64=0.99,
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool=false
)
    # Set random seed if provided or generate one
    if seed === nothing
        seed = rand(1:10000)
    end
    println("Using random seed: $seed")
    Random.seed!(seed)

    # Create a unique directory for this experiment
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    experiment_dir = joinpath(output_dir, "experiment_$(timestamp)")
    if !isdir(experiment_dir)
        mkpath(experiment_dir)
    end

    # Save experiment configuration
    config = Dict(
        "min_end_time" => min_end_time,
        "max_end_time" => max_end_time,
        "solvers" => solvers,
        "num_simulations" => num_simulations,
        "fixed_true_end_time" => fixed_true_end_time,
        "initial_announce" => initial_announce,
        "discount_factor" => discount_factor,
        "seed" => seed,
        "timestamp" => timestamp
    )
    
    config_path = joinpath(experiment_dir, "experiment_config.json")
    open(config_path, "w") do f
        JSON.print(f, config, 4)
    end
    
    if verbose
        println("Experiment configuration saved to: $config_path")
    end

    # Create POMDP
    pomdp = define_pomdp(min_end_time, max_end_time, discount_factor, verbose=verbose, initial_announce=initial_announce, fixed_true_end_time=fixed_true_end_time)
    
    # Generate policies for each solver
    policies = Dict()
    for solver_type in solvers
        policy_data = get_policy(pomdp, solver_type, experiment_dir, verbose=verbose)
        policies[solver_type] = policy_data["policy"]
    end
    
    # Run simulations for all policies
    all_results = Dict()
    
    for sim_num in 1:num_simulations
        if verbose
            println("Running simulation $sim_num of $num_simulations")
        end
        
        # Set seed for reproducibility across policies
        Random.seed!(sim_num)
        
        # Generate true end time for this simulation
        if fixed_true_end_time !== nothing
            true_end_time = fixed_true_end_time
        else
            true_end_time = rand(min_end_time:max_end_time)
        end
        
        # Set initial announced time
        if initial_announce !== nothing
            initial_Ta = initial_announce
        else
            initial_Ta = min_end_time
        end
        

        if verbose
            println("Simulation $sim_num")
        end
        
        # Run simulation for each policy
        for (solver_type, policy) in policies
            if verbose
                println("Running simulation with $solver_type policy")
            end
            
            # Reset RNG to same state for each policy
            Random.seed!(sim_num)
            
            # Run simulation
            sim_trajectory = []
            reward_accumulator = 0.0
            
            for (b, s, a, o, r) in stepthrough(pomdp, policy, DiscreteUpdater(pomdp), "b,s,a,o,r"; max_steps=1_000_000)
                t, Ta, Tt = s
                
                # Update accumulated reward
                reward_accumulator += r
                
                # Record step data
                step_data = Dict(
                    "timestep" => t,
                    "announced_time" => Ta,
                    "true_end_time" => Tt,
                    "action" => a.announced_time,
                    "observation" => o[3],  # Extract the observed time
                    "reward" => r,
                    "accumulated_reward" => reward_accumulator,
                    "belief_estimate" => highest_belief_state(b)[3]  # Get highest probability state's Tt
                )
                
                push!(sim_trajectory, step_data)
                
                # Check if simulation is complete
                if t == Tt
                    break
                end
            end
            
            # Add to results
            if !haskey(all_results, solver_type)
                all_results[solver_type] = []
            end
            push!(all_results[solver_type], sim_trajectory)
        end
    end
    
    # Save all simulation results
    results_path = joinpath(experiment_dir, "experiment_results.json")
    open(results_path, "w") do f
        JSON.print(f, all_results, 4)
    end
    
    if verbose
        println("Experiment results saved to: $results_path")
    end
    
    # Generate plots for each simulation
    for sim_num in 1:num_simulations
        sim_results = Dict(solver => all_results[solver][sim_num] for solver in keys(all_results))
        true_end_time = sim_results[first(keys(sim_results))][1]["true_end_time"]
        
        generate_experiment_plots(
            sim_results, 
            experiment_dir, 
            true_end_time,
            min_end_time,
            max_end_time,
            simulation_number=sim_num
        )
    end
    
    # Generate average performance plots
    generate_average_performance_plots(all_results, experiment_dir)
    
    return experiment_dir, all_results
end

# Generate plots for a single simulation run
function generate_experiment_plots(results, output_dir, true_end_time, min_end_time, max_end_time; simulation_number=1)
    # Create plots directory
    plots_dir = joinpath(output_dir, "plots", "simulation_$simulation_number")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Extract solver types
    solver_types = collect(keys(results))
    
    # Choose colors and markers for each solver
    colors = [:blue, :red, :green, :purple, :orange, :brown, :black, :cyan]
    markers = [:circle, :square, :diamond, :triangle, :cross, :star5, :hexagon, :utriangle]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    # Plot 1: Current announced time vs simulation time
    p1 = plot(
        title = "Announced Time vs Simulation Time (Sim #$simulation_number)",
        xlabel = "Simulation Time (t)",
        ylabel = "Announced Time (Ta)",
        legend = :topleft,
        size = (800, 600),
        grid = true,
        xlims = (0, max_end_time),  # Set x-axis limits
        ylims = (0, max_end_time)   # Set y-axis limits
    )
    
    # Add dashed diagonal line for x=y (current time) up to true end time
    plot!([0, true_end_time], [0, true_end_time], label=nothing, color=:gray, linestyle=:dash, linewidth=1.5)
    
    # Add horizontal line for true end time
    hline!([true_end_time], label="True End Time", color=:black, linewidth=2)
    
    # Add horizontal lines for min and max end times
    hline!([min_end_time], label="Min End Time", color=:red, linestyle=:dash, linewidth=1.5)
    hline!([max_end_time], label="Max End Time", color=:red, linestyle=:dash, linewidth=1.5)
    
    # First, collect all observation points across all solvers
    all_observations = Dict()
    for (solver, trajectory) in results
        for step in trajectory
            t = step["timestep"]
            all_observations[t] = step["observation"]
        end
    end
    
    # Plot observations as a separate series
    timesteps_obs = sort(collect(keys(all_observations)))
    observations = [all_observations[t] for t in timesteps_obs]
    scatter!(
        p1,
        timesteps_obs,
        observations,
        label = "Observations",
        color=:purple,
        marker=:diamond,
        markersize=6,
        markerstrokewidth=0
    )
    
    # Plot announced time trajectory for each solver
    for (i, (solver, trajectory)) in enumerate(results)
        color_idx = mod(i-1, length(colors)) + 1
        marker_idx = mod(i-1, length(markers)) + 1
        style_idx = mod(i-1, length(line_styles)) + 1
        
        # Extract timesteps and announced times (use action values - the announce decisions)
        timesteps = [step["timestep"] for step in trajectory]
        announced_times = [step["action"] for step in trajectory]
        
        # Plot the announced time trajectory
        plot!(
            p1,
            timesteps,
            announced_times,
            label = solver,
            color = colors[color_idx],
            marker = markers[marker_idx],
            markersize = 6,
            linestyle = line_styles[style_idx],
            linewidth = 2
        )
    end
    
    # Save Plot 1
    savefig(p1, joinpath(plots_dir, "announced_time_vs_simulation_time.png"))
    
    # Plot 2: Accumulated reward vs simulation time
    p2 = plot(
        title = "Accumulated Reward vs Simulation Time (Sim #$simulation_number)",
        xlabel = "Simulation Time (t)",
        ylabel = "Accumulated Reward",
        legend = :bottomleft,
        size = (800, 600),
        grid = true,
        xlims = (0, max_end_time)  # Set x-axis limits
    )
    
    # Plot accumulated reward trajectory for each solver
    for (i, (solver, trajectory)) in enumerate(results)
        color_idx = mod(i-1, length(colors)) + 1
        style_idx = mod(i-1, length(line_styles)) + 1
        
        # Extract timesteps and accumulated rewards
        timesteps = [step["timestep"] for step in trajectory]
        accumulated_rewards = [step["accumulated_reward"] for step in trajectory]
        
        # Plot the accumulated reward trajectory
        plot!(
            p2,
            timesteps,
            accumulated_rewards,
            label = solver,
            color = colors[color_idx],
            linestyle = line_styles[style_idx],
            linewidth = 2
        )
    end
    
    # Save Plot 2
    savefig(p2, joinpath(plots_dir, "accumulated_reward_vs_simulation_time.png"))
    
    # Plot 3: Error evolution 
    p3 = generate_error_plot(results, output_dir, simulation_number, min_end_time, max_end_time)
    
    # Combined Plot: Side-by-side (now including 3 plots)
    p_combined = plot(p1, p2, p3, layout = (1, 3), size = (2400, 600))
    savefig(p_combined, joinpath(plots_dir, "combined_plots.png"))
    
    # Return the plot objects in case they're needed
    return p1, p2, p3, p_combined
end

function generate_error_plot(results, output_dir, simulation_number, min_end_time, max_end_time)
    # Create plots directory
    plots_dir = joinpath(output_dir, "plots", "simulation_$simulation_number")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Extract solver types
    solver_types = collect(keys(results))
    
    # Choose colors and markers for each solver
    colors = [:blue, :red, :green, :purple, :orange, :brown, :black, :cyan]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    # Plot error evolution
    p_error = plot(
        title = "Error Evolution (Sim #$simulation_number)",
        xlabel = "Simulation Time (t)",
        ylabel = "Error",
        legend = :topleft,
        size = (800, 600),
        grid = true,
        ylims = (0, max_end_time - min_end_time)  # Set y-axis limits
    )
    
    # Plot error for each solver
    for (i, (solver, trajectory)) in enumerate(results)
        color_idx = mod(i-1, length(colors)) + 1
        style_idx = mod(i-1, length(line_styles)) + 1
        
        # Extract timesteps and errors
        timesteps = [step["timestep"] for step in trajectory]
        errors = [abs(step["action"] - step["true_end_time"]) for step in trajectory]
        
        # Plot the error trajectory
        plot!(
            p_error,
            timesteps,
            errors,
            label = solver,
            color = colors[color_idx],
            linestyle = line_styles[style_idx],
            linewidth = 2
        )
    end
    
    # Save error plot
    savefig(p_error, joinpath(plots_dir, "error_evolution.png"))
    
    return p_error
end

# Generate aggregate performance plots across all simulations
function generate_average_performance_plots(all_results, output_dir)
    # Create plots directory
    plots_dir = joinpath(output_dir, "plots", "average_performance")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Extract solver types
    solver_types = collect(keys(all_results))
    
    # Compute average final rewards for each solver
    avg_rewards = Dict()
    mean_avg_error = Dict()
    mean_max_error = Dict()
    for solver in solver_types
        simulations = all_results[solver]
        avg_error = []
        max_error = []
        #print(simulations)
        for sim in simulations
            actions = [sim[i]["action"] for i in eachindex(sim)]
            true_end_times = [sim[i]["true_end_time"] for i in eachindex(sim)]
            error = abs.(actions-true_end_times)
            push!(avg_error,mean(error))
            push!(max_error,maximum(error))
        end
        final_rewards = [sim[end]["accumulated_reward"] for sim in simulations]
        avg_rewards[solver] = mean(final_rewards)
        mean_avg_error[solver] = mean(avg_error)
        mean_max_error[solver] = mean(max_error)        
    end
    
    # Sort solvers by average reward
    sorted_solvers = sort(collect(keys(avg_rewards)), by=s -> avg_rewards[s], rev=true)
    
    # Plot average final rewards
    p_avg = bar(
        sorted_solvers,
        [avg_rewards[s] for s in sorted_solvers],
        title = "Average Final Reward by Solver",
        xlabel = "Solver",
        ylabel = "Average Final Reward",
        legend = false,
        size = (800, 600),
        grid = true,
        rotation = 45,  # Rotate x-axis labels
        bottom_margin = 10Plots.mm  # Add margin for rotated labels
    )
    
    # Add value labels above bars
    for (i, solver) in enumerate(sorted_solvers)
        annotate!(i, avg_rewards[solver] + maximum(values(avg_rewards))*0.03, 
                 text(round(avg_rewards[solver], digits=1), 8, :center))
    end
    
    savefig(p_avg, joinpath(plots_dir, "average_final_reward.png"))
    
    # Sort solvers by average error
    sorted_solvers = sort(collect(keys(mean_avg_error)), by=s -> mean_avg_error[s], rev=true)
    
    # Plot average errors
    e_avg = bar(
        sorted_solvers,
        [mean_avg_error[s] for s in sorted_solvers],
        title = "Average Mean Error by Solver",
        xlabel = "Solver",
        ylabel = "Average Mean Error",
        legend = false,
        size = (800, 600),
        grid = true,
        rotation = 45,  # Rotate x-axis labels
        bottom_margin = 10Plots.mm  # Add margin for rotated labels
    )
    
    # Add value labels above bars
    for (i, solver) in enumerate(sorted_solvers)
        annotate!(i, mean_avg_error[solver] + maximum(values(mean_avg_error))*0.03, 
                 text(round(mean_avg_error[solver], digits=1), 8, :center))
    end
    
    savefig(e_avg, joinpath(plots_dir, "average_mean_error.png"))

    # Sort solvers by average max error
    sorted_solvers = sort(collect(keys(mean_max_error)), by=s -> mean_max_error[s], rev=true)
    
    # Plot average max errors
    m_avg = bar(
        sorted_solvers,
        [mean_max_error[s] for s in sorted_solvers],
        title = "Average Max Error by Solver",
        xlabel = "Solver",
        ylabel = "Average Max Error",
        legend = false,
        size = (800, 600),
        grid = true,
        rotation = 45,  # Rotate x-axis labels
        bottom_margin = 10Plots.mm  # Add margin for rotated labels
    )
    
    # Add value labels above bars
    for (i, solver) in enumerate(sorted_solvers)
        annotate!(i, mean_max_error[solver] + maximum(values(mean_max_error))*0.03, 
                 text(round(mean_max_error[solver], digits=1), 8, :center))
    end
    
    savefig(m_avg, joinpath(plots_dir, "average_max_error.png"))

    return p_avg, e_avg, m_avg
end