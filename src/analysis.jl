function create_evaluation_plots(stats, output_dir)
    # Make sure plots directory exists
    plots_dir = joinpath(output_dir, "plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Plot reward distribution
    p1 = histogram(stats["rewards"], 
                 bins=20, 
                 title="Reward Distribution",
                 xlabel="Total Reward",
                 ylabel="Frequency",
                 legend=false,
                 fillalpha=0.7,
                 color=:blue)
    
    # Add mean and median lines
    reward_mean = mean(stats["rewards"])
    reward_median = median(stats["rewards"])
    vline!([reward_mean], label="Mean", linewidth=2, color=:red)
    vline!([reward_median], label="Median", linewidth=2, color=:green, linestyle=:dash)
    
    savefig(p1, joinpath(plots_dir, "reward_distribution.png"))
    
    # Plot error metrics
    p2 = plot(title="Error Metrics",
            xlabel="Metric",
            ylabel="Value",
            legend=false,
            xticks=(1:3, ["Initial Error", "Final Error", "Change Magnitude"]),
            grid=false,
            boxplot=true)
    
    boxplot!(p2, [1], stats["initial_errors"], fillalpha=0.7, color=:blue)
    boxplot!(p2, [2], stats["final_errors"], fillalpha=0.7, color=:red)
    boxplot!(p2, [3], stats["avg_change_magnitudes"], fillalpha=0.7, color=:green)
    
    savefig(p2, joinpath(plots_dir, "error_metrics.png"))
    
    # Plot number of changes
    p3 = histogram(stats["num_changes"],
                 bins=maximum(stats["num_changes"]) - minimum(stats["num_changes"]) + 1,
                 title="Number of Announcement Changes",
                 xlabel="Number of Changes",
                 ylabel="Frequency",
                 legend=false,
                 fillalpha=0.7,
                 color=:purple)
    
    savefig(p3, joinpath(plots_dir, "num_changes.png"))
        
    return [p1, p2, p3]
end

# Add these functions to src/analysis.jl

function plot_belief_distribution(belief, true_end_time, min_end_time, max_end_time, timestep, announced_time; title_prefix="", is_momdp=false)
    states, probs = extract_belief_states_and_probs(belief)
    
    # Extract only the Tt (true end time) component and its probability
    end_time_probs = Dict{Int, Float64}()
    
    # Aggregate probabilities by end time (may have multiple states with same end time)
    for (state, prob) in zip(states, probs)
        if is_momdp == true
            Tt = state[2]  # Extract true end time from state tuple
        else
            # For standard POMDP, state is a tuple (t, Ta, Tt)
            Tt = state[3]
        end
        end_time_probs[Tt] = get(end_time_probs, Tt, 0.0) + prob
    end
    
    # Create x-axis with all possible end times
    x_values = collect(min_end_time:max_end_time)
    y_values = [get(end_time_probs, x, 0.0) for x in x_values]
    
    p = bar(
        x_values,
        y_values,
        title="$(title_prefix)Belief Distribution at t=$(timestep)",
        xlabel="Possible End Time",
        ylabel="Probability",
        legend=false,
        fillalpha=0.7,
        color=:blue,
        size=(800, 400),
        xticks=(x_values, x_values)
    )
    
    # Add vertical line for true end time
    vline!([true_end_time], label="True End Time", linewidth=2, color=:red, linestyle=:dash)
    vline!([announced_time], label="Announced End Time", linewidth=2, color=:black, linestyle=:dash)
    # Add the highest belief state as text annotation
    highest_end_time = x_values[argmax(y_values)]
    # annotate!(highest_end_time, maximum(y_values) * 1.05, 
    #           text("Highest Belief: $highest_end_time", 10, :center))
    
    return p
end

function plot_announce_time_evolution(run_details, true_end_time, min_end_time, max_end_time; 
                                     title_prefix="", include_observations=true)
    # Extract data
    timesteps = [step["timestep"] for step in run_details]
    announced_times = [step["action"] for step in run_details]
    
    p = plot(
        title="$(title_prefix)Announced Time vs Simulation Time",
        xlabel="Simulation Time (t)",
        ylabel="Announced Time",
        legend=:topleft,
        size=(800, 500),
        grid=true,
        xlims = (0, max_end_time),  # Set x-axis limits
        ylims = (0, max_end_time)   # Set y-axis limits
    )
    
    # Add dashed diagonal line for t=y (current time) up to true end time
    plot!([0, true_end_time], [0, true_end_time], label=nothing, color=:gray, linestyle=:dash, linewidth=1.5)
    
    # Add horizontal line for true end time
    hline!([true_end_time], label=nothing, color=:black, linewidth=2)
    
    # Add horizontal lines for min and max end times
    hline!([min_end_time], label=nothing, color=:red, linestyle=:dash, linewidth=1.5)
    hline!([max_end_time], label=nothing, color=:red, linestyle=:dash, linewidth=1.5)
    
    # Plot the announced time trajectory
    plot!(
        timesteps,
        announced_times,
        label="Announced Time",
        color=:blue,
        marker=:circle,
        markersize=6,
        linewidth=2
    )
    
    # Include observations if requested
    if include_observations
        observations = [step["To"] for step in run_details]
        scatter!(
            timesteps,
            observations,
            label="Observations",
            color=:purple,
            marker=:diamond,
            markersize=6,
            markerstrokewidth=0
        )
    end
    
    return p
end

function plot_reward_evolution(run_details; title_prefix="")
    # Extract data
    timesteps = [step["timestep"] for step in run_details]
    rewards = [step["reward"] for step in run_details]
    cumulative_rewards = [step["cumulative_reward"] for step in run_details]
    
    p = plot(
        title="$(title_prefix)Reward Evolution",
        xlabel="Simulation Time (t)",
        ylabel="Reward",
        legend=:bottomleft,
        size=(800, 500),
        grid=true
    )
    
    # Plot step rewards
    plot!(
        timesteps,
        rewards,
        label="Step Reward",
        color=:green,
        marker=:circle,
        markersize=4,
        linewidth=1,
        linealpha=0.5
    )
    
    # Plot cumulative rewards on secondary y-axis
    plot!(
        twinx(),
        timesteps,
        cumulative_rewards,
        label="Cumulative Reward",
        color=:blue,
        linewidth=2,
        ylabel="Cumulative Reward"
    )
    
    return p
end

function plot_observation_probability(pomdp, state, true_end_time, min_end_time, max_end_time, actual_observation=nothing; title_prefix="", is_momdp=false)
    t, Ta, Tt = state
    
    # Skip if at or past end time
    if t >= Tt
        return nothing
    end
    
    # Create dummy action to use in observation model
    if is_momdp
        # For MOMDP, we need to adjust the state tuple
        a = Ta
        next_state = ((t+1, Ta), Tt)
    else
        # For standard POMDP, we keep the same state structure
        a = AnnounceAction(Ta)
        next_state = (t+1, Ta, Tt)
    end
    
    # Get observation distribution
    obs_dist = POMDPs.observation(pomdp, a, next_state)
    
    # If it's a deterministic distribution, convert to histogram format
    if obs_dist isa Deterministic
        o = obs_dist.val
        if is_momdp
            # For MOMDP, the observation is a tuple (t, Ta, To)
            obs_time = o # Extract To from the tuple
        else
            # For standard POMDP, the observation is a tuple (t, Ta, Tt)    
            obs_time = o[3]
        end
        x_values = collect(min_end_time:max_end_time)
        y_values = zeros(length(x_values))
        idx = findfirst(x -> x == obs_time, x_values)
        if idx !== nothing
            y_values[idx] = 1.0
        end
    else
        # For SparseCat distribution
        obs_list = obs_dist.vals
        probs = obs_dist.probs
        
        # Extract only the To (observed time) component
        if is_momdp
            obs_times = [o for o in obs_list]
        else
            obs_times = [o[3] for o in obs_list]
        end
        
        # Create mapping of observation time to probability
        time_probs = Dict{Int, Float64}()
        for (obs, prob) in zip(obs_times, probs)
            time_probs[obs] = get(time_probs, obs, 0.0) + prob
        end
        
        # Create vectors for plotting
        x_values = collect(min_end_time:max_end_time)
        y_values = [get(time_probs, x, 0.0) for x in x_values]
    end
    
    p = bar(
        x_values,
        y_values,
        title="$(title_prefix)Observation Probability at t=$(t)",
        xlabel="Possible Observed End Time",
        ylabel="Probability",
        legend=true,
        fillalpha=0.7,
        color=:purple,
        size=(800, 400),
        xticks=(x_values, x_values)  # Set ticks at integer positions
    )
    
    # Add vertical line for true end time
    vline!([true_end_time], label="True End Time", linewidth=2, color=:red, linestyle=:dash)
    
    # Add vertical line for actual observation if provided
    if actual_observation !== nothing
        vline!([actual_observation], label="Actual Observation", linewidth=2, color=:green, linestyle=:dash)
    end
    
    return p
end

function plot_error_evolution(run_details; title_prefix="")
    # Extract data
    timesteps = [step["timestep"] for step in run_details]
    announced_errors = [step["announced_error"] for step in run_details]
    belief_errors = [step["belief_error"] for step in run_details]
    
    p = plot(
        title="$(title_prefix)Error Evolution",
        xlabel="Simulation Time (t)",
        ylabel="Error",
        legend=:topleft,
        size=(800, 500),
        grid=true
    )
    
    # Plot announced error
    plot!(
        timesteps,
        announced_errors,
        label="Announced Error",
        color=:red,
        marker=:circle,
        markersize=4,
        linewidth=2
    )
    
    # Plot belief error
    plot!(
        timesteps,
        belief_errors,
        label="Belief Error",
        color=:blue,
        marker=:diamond,
        markersize=4,
        linewidth=2
    )
    
    return p
end

function plot_2d_belief_evolution(belief_history, true_end_time, min_end_time, max_end_time; title_prefix="", is_momdp=false)
    """
    Creates a 2D heatmap showing the evolution of belief probabilities over time.
    
    Args:
        belief_history: Vector of belief states from simulation
        true_end_time: The actual true end time for this simulation
        min_end_time: Minimum possible end time in the problem
        max_end_time: Maximum possible end time in the problem
        title_prefix: Optional prefix for the plot title
    
    Returns:
        Plots.Plot object containing the heatmap
    """
    
    if belief_history === nothing || isempty(belief_history)
        @warn "No belief history available for 2D belief evolution plot"
        return nothing
    end
    
    num_timesteps = length(belief_history)
    possible_end_times = collect(min_end_time:max_end_time)
    num_end_times = length(possible_end_times)
    
    # Initialize probability matrix: rows = end times, columns = timesteps
    prob_matrix = zeros(Float64, num_end_times, num_timesteps)
    
    # Fill the probability matrix
    for (timestep_idx, belief) in enumerate(belief_history)
        states, probs = extract_belief_states_and_probs(belief)
        
        # Aggregate probabilities by true end time (Tt)
        end_time_probs = Dict{Int, Float64}()
        for (state, prob) in zip(states, probs)
            if is_momdp
                Tt = state[2]  # Extract true end time from state tuple
            else
                # For standard POMDP, state is a tuple (t, Ta, Tt)
                Tt = state[3]
            end
            end_time_probs[Tt] = get(end_time_probs, Tt, 0.0) + prob
        end
        
        # Fill the matrix column for this timestep
        for (end_time_idx, end_time) in enumerate(possible_end_times)
            prob_matrix[end_time_idx, timestep_idx] = get(end_time_probs, end_time, 0.0)
        end
    end
    
    # Create timestep labels (starting from 0)
    timestep_labels = collect(0:(num_timesteps-1))
    
    # Create the heatmap
    p = heatmap(
        timestep_labels,
        possible_end_times,
        prob_matrix,
        title = "$(title_prefix)2D Belief Evolution Over Time",
        xlabel = "Simulation Time (t)",
        ylabel = "True End Time (Tt)",
        color = :viridis,
        aspect_ratio = :auto,
        size = (800, 600),
        colorbar_title = "Probability",
        legend = :outertopright  # Place legend outside the plot on the top right
    )
    
    # Add a horizontal line for the actual true end time
    hline!([true_end_time], 
           label = "True End Time", 
           color = :red, 
           linewidth = 3, 
           linestyle = :dash,
           legend = :topright)

    # Ensure proper tick spacing for readability
    plot!(
        xticks = (0:2:maximum(timestep_labels), 0:2:maximum(timestep_labels)),
        yticks = (min_end_time:2:max_end_time, min_end_time:2:max_end_time)
    )
    
    return p
end

function plot_2d_belief_evolution_with_actions(belief_history, run_details, true_end_time, min_end_time, max_end_time; title_prefix="", is_momdp=false)
    """
    Enhanced version that also overlays the announced times as a trajectory.
    """
    
    # Create the base 2D belief evolution plot
    p = plot_2d_belief_evolution(belief_history, true_end_time, min_end_time, max_end_time, title_prefix=title_prefix, is_momdp=is_momdp)
    
    if p === nothing
        return nothing
    end
    
    # Extract timesteps and announced times from run details
    timesteps = [step["timestep"] for step in run_details]
    announced_times = [step["action"] for step in run_details]
    
    # Overlay the announced time trajectory
    plot!(p,
        timesteps,
        announced_times,
        label = "Announced Time",
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
        scatter!(p,
            timesteps,
            observations,
            label = "Observations",
            color = :yellow,
            marker = :diamond,
            markersize = 5,
            markerstrokecolor = :black,
            markerstrokewidth = 1
        )
    end
    
    return p
end

function create_debug_plots(pomdp, run_details, min_end_time, max_end_time, output_dir;
                           belief_history=nothing, plot_beliefs=true, plot_observations=true)
    # Make sure plots directory exists
    plots_dir = joinpath(output_dir, "debug_plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end

    if isa(pomdp, PlanningProblem)
        is_momdp = true
    else
        is_momdp = false
    end
    
    # Extract true end time from the first step
    true_end_time = run_details[1]["Tt"]

    announced_times = [step["action"] for step in run_details]
    
    # Plot announce time evolution
    p_announce = plot_announce_time_evolution(
        run_details, 
        true_end_time, 
        min_end_time, 
        max_end_time
    )
    savefig(p_announce, joinpath(plots_dir, "announce_time_evolution.png"))
    
    # Plot reward evolution
    p_reward = plot_reward_evolution(run_details)
    savefig(p_reward, joinpath(plots_dir, "reward_evolution.png"))

    # Plot error evolution
    p_error = plot_error_evolution(run_details)
    savefig(p_error, joinpath(plots_dir, "error_evolution.png"))
    
    # Plot belief distributions if available
    if belief_history !== nothing && plot_beliefs
        belief_plots_dir = joinpath(plots_dir, "beliefs")
        if !isdir(belief_plots_dir)
            mkpath(belief_plots_dir)
        end
        
        for (i, belief) in enumerate(belief_history)
            timestep = i - 1  # Timestep starts at 0
            
            p_belief = plot_belief_distribution(
                belief, 
                true_end_time, 
                min_end_time, 
                max_end_time, 
                timestep,
                announced_times[i],
                is_momdp=is_momdp
            )
            savefig(p_belief, joinpath(belief_plots_dir, "belief_t$(lpad(timestep, 2, '0')).png"))
        end
    end
    
    # Plot observation probabilities if requested
    if plot_observations
        obs_plots_dir = joinpath(plots_dir, "observations")
        if !isdir(obs_plots_dir)
            mkpath(obs_plots_dir)
        end
        
        for (i, step) in enumerate(run_details)
            # Skip the last step
            if i == length(run_details)
                continue
            end
            
            timestep = step["timestep"]
            Ta = step["Ta_prev"]
            Tt = step["Tt"]
            
            # Create state tuple
            state = (timestep, Ta, Tt)
            
            p_obs = plot_observation_probability(
                pomdp,
                state,
                true_end_time,
                min_end_time,
                max_end_time,
                step["To"],
                is_momdp=is_momdp
            )
            
            if p_obs !== nothing
                savefig(p_obs, joinpath(obs_plots_dir, "obs_prob_t$(lpad(timestep, 2, '0')).png"))
            end
        end
    end

    # Plot 2D belief evolution
    p_belief_2d = plot_2d_belief_evolution(
        belief_history, 
        true_end_time, 
        min_end_time, 
        max_end_time,
        is_momdp=is_momdp
    )
    if p_belief_2d !== nothing
        savefig(p_belief_2d, joinpath(plots_dir, "belief_evolution_2d.png"))
    end
    
    p_belief_2d_enhanced = plot_2d_belief_evolution_with_actions(
        belief_history,
        run_details,
        true_end_time, 
        min_end_time, 
        max_end_time,
        is_momdp=is_momdp
    )
    if p_belief_2d_enhanced !== nothing
        savefig(p_belief_2d_enhanced, joinpath(plots_dir, "belief_evolution_2d_with_actions.png"))
    end
    
    # Return the main plots
    return p_announce, p_reward
end