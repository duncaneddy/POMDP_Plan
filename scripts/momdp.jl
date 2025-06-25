using POMDPs
using POMDPTools
using SARSOP
using MOMDPs

# Utilities
using Statistics
using Distributions
using Random
using LinearAlgebra
using JSON
using JLD2
using Dates
using ProgressMeter
using Plots
using StatsPlots

# Hack for evaluation
using POMDPPlanning: highest_belief_state, extract_belief_states_and_probs

###
# Background
###

# Setup
# using Pkg
# Pkg.add(["POMDPs", "POMDPTools", "SARSOP", "MOMDPs", "Distributions", "Random", "LinearAlgebra", "JSON", "JLD2", "Dates", "ProgressMeter", "Plots", "StatsPlots"])

# This file presents the problem of task planning in a MOMDP framework.
# The problem is to plan a project with a true end time Tt, which is unknown
# to the planner. 
#
# At each time step the planner can announce an end time Ta, which is a guess
# of the true end time Tt. The planner receives noisy observations of the true end time
# Tt at each time step t, and the goal is to maximize the reward by minimizing
# the difference between the announced end time Ta and the true end time Tt.
#
# The reward function is very simple. It is just the difference between
# the announced end time Ta and the true end time Tt.


###
# Define the PlanningProblem MOMDP
###

mutable struct PlanningProblem <: MOMDP{Tuple{Int, Int}, Int, Int, Int}
    min_end_time::Int
    max_end_time::Int
    discount_factor::Float64
    initial_announced_time::Union{Int, Nothing}
end

# Define these relationships for the MOMDP to improve performance
MOMDPs.is_y_prime_dependent_on_x_prime(::PlanningProblem) = false
MOMDPs.is_x_prime_dependent_on_y(::PlanningProblem) = true
MOMDPs.is_initial_distribution_independent(::PlanningProblem) = true

# Define core MOMDP functions

# The observable states are tuples of (t, Ta), where:
# - `t` is the current time step (0 to max_end_time)
# - `Ta` is the announced end time (from min_end_time to max_end_time
function MOMDPs.states_x(problem::PlanningProblem)
    # Visible state: (t, Ta)
    return vcat([(t, Ta) for t in 0:problem.max_end_time, Ta in problem.min_end_time:problem.max_end_time]...)
end

function MOMDPs.states_y(problem::PlanningProblem)
    # Hidden state: Tt (true end time)
    return [Tt for Tt in problem.min_end_time:problem.max_end_time]
end

# Variant for (x, y) states
function MOMDPs.stateindex_x(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int})
    findfirst(s -> s == state[1], states_x(problem))
end

# Variant for (x) states
function MOMDPs.stateindex_x(problem::PlanningProblem, state::Tuple{Int, Int})
    findfirst(s -> s == state, states_x(problem))
end

# Variant for (x, y) states
function MOMDPs.stateindex_y(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int})
    findfirst(s -> s == state[2], states_y(problem))
end

# Variant for (y) states
function MOMDPs.stateindex_y(problem::PlanningProblem, state::Int)
    findfirst(s -> s == state, states_y(problem))
end

# Provides the probability distribution over the fully observable states at the initial timestep.
function MOMDPs.initialstate_x(problem::PlanningProblem)
    if problem.initial_announced_time !== nothing
        return Deterministic((0, problem.initial_announced_time))
    else
        return Deterministic((0, problem.min_end_time))  # Default to min_end_time if not specified
    end
end

# Given a fully observable initial state x, this function returns the initial probability distribution over the partially observable states.
function MOMDPs.initialstate_y(problem::PlanningProblem, xprime::Tuple{Int, Int})
    num_states_y = length(states_y(problem))
    return SparseCat(states_y(problem), fill(1.0 / num_states_y, num_states_y))
end

# Provides the distribution over the next fully observable state x' given the current state (x,y) and action a.
function MOMDPs.transition_x(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int)
    # If the time is already at or past the true end time Tt, return the same state
    # Or if the incremented time will be past the maximum end time return the same state
    if state[1][1] >= state[2] || state[1][1] + 1 > problem.max_end_time
        # If the project is done, return the same state
        return Deterministic(state[1])
        # return Deterministic(state) # This is incorrect type of Tuple{Tuple{Int, Int}, Int}
    else
        # Increment the time and return the new state with the action
        return Deterministic((state[1][1] + 1, action))
    end
end

function MOMDPs.transition_y(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int, xprime::Tuple{Int, Int})
    # The true end time Tt is independent of the action, so we return the same state
    return Deterministic(state[2])
end

## Define additional helpers for the MOMDP

function POMDPs.discount(problem::PlanningProblem)
    return problem.discount_factor
end

function POMDPs.actions(problem::PlanningProblem)
    return problem.min_end_time:problem.max_end_time
end

function POMDPs.actions(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int})
    # Given a state ((t, Ta), Tt) the actions are to pick a time in the range of [t, problem.max_end_time]
    t, Ta = state[1]
    return t:problem.max_end_time
end

function POMDPs.actionindex(problem::PlanningProblem, action::Int)
    return findfirst(a -> a == action, POMDPs.actions(problem))
end

function POMDPs.observations(problem::PlanningProblem)
    # The observation is the true end time Tt, which can be any value in the range of [problem.min_end_time, problem.max_end_time]
    return collect(problem.min_end_time:problem.max_end_time)
end

function POMDPs.obsindex(problem::PlanningProblem, observation::Int)
    # The observation is the true end time Tt, which can be any value in the range of [problem.min_end_time, problem.max_end_time]
    return findfirst(o -> o == observation, POMDPs.observations(problem))
end

function POMDPs.observation(problem::PlanningProblem, action::Int, state::Tuple{Tuple{Int, Int}, Int})
    t, Ta = state[1]
    Tt = state[2]

    # If the project is done return the true end time Tt
    if t >= Tt || t + 1 == problem.max_end_time
        return Deterministic(Tt)
    end

    min_obs_time = max(t + 1, problem.min_end_time)

    possible_Tos = collect(min_obs_time:problem.max_end_time)

    # Calculate parameters of the truncated normal
    mu = Tt
    std = (Tt - t) / 3 # MAGIC NUMBER
    
    if Tt - t <= 0 
        # The task is done, so we should observe the true end time
        return Deterministic(Tt)
    end

    base_dist = Normal(mu, std)

    # the truncated normal distribution is defined from t+1 to max_end_time
    lower = min_obs_time
    upper = problem.max_end_time
    cdf_lower = cdf(base_dist, lower)
    cdf_upper = cdf(base_dist, upper)
    denom = cdf_upper - cdf_lower

    probs = Float64[]
    for To_val in possible_Tos
        p = (pdf(base_dist, To_val) / denom)
        push!(probs, p)
    end

    total_p = sum(probs)
    if total_p == 0.0
        return Deterministic(Tt)
    end
    probs ./= total_p
    
    return SparseCat(possible_Tos, probs)

end

function POMDPs.reward(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int)
    t, Ta = state[1]
    Tt = state[2]

    # If the project is done, return a reward of 1
    if t >= Tt || t + 1 == problem.max_end_time
        return 0.0 # Originally 1.0
    end

    # Dead-simple reward - just penalize for the difference between announced and true end time
    # return -1 * abs(Ta - Tt)

    # Dead-simple reward - just penalize for the difference between announced and true end time
    r = -2 * abs(action - Tt)

    # Add penalty if action changes from previous announced time
    if t > 0 && Ta != action
        r -= 3.0  # Penalty for changing the announced time
    end

    if Tt == t
        # if Tt = t, we must announce t = Tt
        if action != Tt
            r -= 1000
        end
    end

    return r
end

function POMDPs.isterminal(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int})
    # The project is done if the current time t is greater than or equal to the true end time Tt
    return state[1][1] >= state[2]
end

####
# Hack together evaluation code
####

function simulate_single(momdp, policy; 
                        fixed_true_end_time=nothing,
                        initial_announce=nothing,
                        collect_beliefs=true,
                        verbose=false)
    step = 0
    r_sum = 0
    obs_old = NaN # dummy initialization
    iteration_details = []
    belief_history = collect_beliefs ? [] : nothing

    # Track metrics
    announcement_changes = []
    first_announced = nothing

    # Get domain parameters from POMDP
    true_end_times = states_y(momdp)
    min_end_time = minimum(true_end_times)
    max_end_time = maximum(true_end_times)

    # Set up initial state with fixed true end time if specified
    if fixed_true_end_time !== nothing
        if fixed_true_end_time < min_end_time || fixed_true_end_time > max_end_time
            @warn "Fixed true end time $fixed_true_end_time outside valid range [$min_end_time, $max_end_time]"
            fixed_true_end_time = nothing
        end
    end

    # updater = MOMDPDiscreteUpdater(momdp)

    updater = DiscreteUpdater(momdp)

    for (b, s, a, o, r) in stepthrough(momdp, policy, updater, "b,s,a,o,r"; max_steps=1_000_000) # should be able to set max_steps=max_end_time+1
        # println("Step: $step, Belief: $b, State: $s Action: $a, Observation: $o, Reward: $r")
        # println("Step: $step, State: $s Action: $a, Observation: $o, Reward: $r")
        r_sum += r
        step += 1
        t, Ta = s[1]
        Tt = s[2]
        

        # Store belief for later plotting if requested
        if collect_beliefs
            push!(belief_history, deepcopy(b))
        end

        # Record first announced time
        if step == 1
            first_announced = Ta
        end

        # Track announcement changes
        if step > 1 && Ta != a
            push!(announcement_changes, (t, Ta, a, a - Ta))
        end

        if verbose
            println("Timestep: ", t)
            println("True End Time: ", Tt)
            println("Previous Announced Time: ", Ta)
            println("Old Observation: ", obs_old)
            println("Action: ", a)
            println("Observed Time: ", o)
            obs_old = o
            # print_belief_states_and_probs(b)
            @show r r_sum
            println()
        end

        # Save detailed metrics for this step
        iteration_detail = Dict(
            "timestep" => t,
            "Tt" => Tt,
            "Ta_prev" => Ta,
            "To_prev" => obs_old,
            "action" => a,
            "To" => o,
            "reward" => r,
            "cumulative_reward" => r_sum,
            "high_b_Tt" => highest_belief_state(b)[2],
            "belief_error" => abs(highest_belief_state(b)[2] - Tt),
            "announced_error" => abs(a - Tt)
        )
        push!(iteration_details, iteration_detail)

        obs_old = o
        if t == Tt
            if verbose
                println("Project complete!")
            end
            break
        end
    end

    # Calculate the requested metrics
    final_iter = iteration_details[end]

    # Initial error (first announced vs true end time)
    initial_error = abs(first_announced - final_iter["Tt"])

    # Final error (final announced vs true end time)
    final_error = abs(final_iter["action"] - final_iter["Tt"])

    # Is final announce time less than true end time (undershoot=true, overshoot=false)
    final_undershoot = final_iter["action"] < final_iter["Tt"]

    # Number of announcement changes
    num_changes = length(announcement_changes)

    # Average magnitude of announcement changes
    avg_change_magnitude = num_changes > 0 ? mean([abs(change[4]) for change in announcement_changes]) : 0

    # Standard deviation of announcement changes
    std_change_magnitude = num_changes > 0 ? std([change[4] for change in announcement_changes]) : 0

    # Create metrics dictionary
    metrics = Dict(
        "initial_error" => initial_error,
        "final_error" => final_error,
        "final_undershoot" => final_undershoot,
        "num_changes" => num_changes,
        "announcement_changes" => announcement_changes,
        "avg_change_magnitude" => avg_change_magnitude, 
        "std_change_magnitude" => std_change_magnitude,
        "total_reward" => r_sum,
        "iterations" => iteration_details,
        "belief_history" => belief_history,
        "min_end_time" => min_end_time,
        "max_end_time" => max_end_time
    )

    return metrics
end

# Consolidated function to run multiple simulations
function simulate_many(momdp, policy, num_simulations;
                    fixed_true_end_time=nothing,
                    initial_announce=nothing,
                    collect_beliefs=false,
                    seed=nothing,
                    verbose=false)

    # Set random seed if provided
    if seed !== nothing
        println("Setting random seed to $seed")
        Random.seed!(seed)
    end

    println("Running $num_simulations simulation(s)")

    # Collect metrics across all simulations
    rewards = Float64[]
    initial_errors = Int[]
    final_errors = Int[]
    final_undershoot = Bool[]
    num_changes = Int[]
    avg_change_magnitudes = Float64[]
    std_change_magnitudes = Float64[]
    all_run_details = []
    simulation_metrics = []

    progress = Progress(num_simulations, desc="Running simulations...")

    for i in 1:num_simulations
        if verbose && i % 10 == 0
            println("Simulation $i of $num_simulations")
        end

        # Run a single simulation and collect metrics
        metrics = simulate_single(
            momdp, 
            policy,
            fixed_true_end_time=fixed_true_end_time,
            initial_announce=initial_announce,
            collect_beliefs=collect_beliefs,
            verbose=(verbose && i == 1) # Only show verbose output for first simulation
        )

        # Store metrics from this run
        push!(rewards, metrics["total_reward"])
        push!(initial_errors, metrics["initial_error"])
        push!(final_errors, metrics["final_error"])
        push!(final_undershoot, metrics["final_undershoot"])
        push!(num_changes, metrics["num_changes"])
        push!(avg_change_magnitudes, metrics["avg_change_magnitude"])
        push!(std_change_magnitudes, metrics["std_change_magnitude"])
        push!(all_run_details, metrics["iterations"])
        push!(simulation_metrics, metrics)

        update!(progress, i)
    end

    # Compile aggregated statistics
    stats = Dict(
        "num_simulations" => num_simulations,
        "rewards" => rewards,
        "initial_errors" => initial_errors,
        "final_errors" => final_errors,
        "final_undershoot" => final_undershoot,
        "num_changes" => num_changes,
        "avg_change_magnitudes" => avg_change_magnitudes,
        "std_change_magnitudes" => std_change_magnitudes,
        "run_details" => all_run_details,
        "seed" => seed,
        "simulation_metrics" => simulation_metrics,
        "min_end_time" => simulation_metrics[1]["min_end_time"],
        "max_end_time" => simulation_metrics[1]["max_end_time"]
    )

    return stats
end

function evaluate_policy(momdp, policy, num_simulations, output_dir;
                        fixed_true_end_time=nothing,
                        initial_announce=nothing,
                        seed=nothing,
                        verbose=false,
                        solver="UnknownSolver")

    println("Evaluating policy for $num_simulations simulation(s)")

    # Set random seed if provided
    if seed !== nothing
        println("Setting random seed to $seed")
        Random.seed!(seed)
    else
        # Generate a random seed
        seed = rand(1:10000)
        println("Using random seed: $seed")
        Random.seed!(seed)
    end

    # Prepare output directory
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Create a timestamp for this evaluation run
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    eval_dir = joinpath(output_dir, "evaluation_$(timestamp)")
    if !isdir(eval_dir)
        mkpath(eval_dir)
    end

    # Save evaluation parameters
    params = Dict(
        "num_simulations" => num_simulations,
        "fixed_true_end_time" => fixed_true_end_time,
        "initial_announce" => initial_announce,
        "seed" => seed,
        "timestamp" => timestamp,
        "solver" => solver,
    )

    params_path = joinpath(eval_dir, "evaluation_params.json")
    open(params_path, "w") do f
        JSON.print(f, params, 4)
    end

    # Run all simulations
    stats = simulate_many(
        momdp, 
        policy, 
        num_simulations,
        fixed_true_end_time=fixed_true_end_time,
        initial_announce=initial_announce,
        collect_beliefs=true, # Always collect beliefs for evaluations
        seed=seed,
        verbose=verbose
    )

    # Generate debug plots for each simulation
    plots_base_dir = joinpath(eval_dir, "simulation_plots")
    if !isdir(plots_base_dir)
        mkpath(plots_base_dir)
    end

    println("Generating debug plots for each simulation...")
    progress = Progress(num_simulations, desc="Creating plots...")

    for i in 1:num_simulations
        sim_metrics = stats["simulation_metrics"][i]
        sim_dir = joinpath(plots_base_dir, "simulation_$(lpad(i, 3, '0'))")

        if !isdir(sim_dir)
            mkpath(sim_dir)
        end

        # Create debug plots for this simulation
        create_debug_plots(
            momdp,
            sim_metrics["iterations"],
            sim_metrics["min_end_time"],
            sim_metrics["max_end_time"],
            sim_dir,
            belief_history=sim_metrics["belief_history"],
            plot_beliefs=true,
            plot_observations=true
        )

        update!(progress, i)
    end

    # Save aggregated evaluation results
    results_filepath = joinpath(eval_dir, "evaluation_results.json")
    open(results_filepath, "w") do f
        # Remove belief histories before saving to JSON to keep file size reasonable
        save_stats = deepcopy(stats)
        for metrics in save_stats["simulation_metrics"]
            delete!(metrics, "belief_history")
        end
        JSON.print(f, save_stats, 4)
    end

    if verbose
        println("Evaluation results saved to: $results_filepath")
    end

    # Print summary statistics
    println("\nEvaluation Summary:")
    println("-------------------")
    println("Random seed: $(seed)")
    println("Average reward: $(mean(stats["rewards"]))")
    println("Min reward: $(minimum(stats["rewards"]))")
    println("Max reward: $(maximum(stats["rewards"]))")
    println("Average number of announcement changes: $(mean(stats["num_changes"]))")
    println("Average initial error: $(mean(stats["initial_errors"]))")
    println("Average final error: $(mean(stats["final_errors"]))")
    println("Percentage of runs with final undershoot: $(mean(stats["final_undershoot"]) * 100)%")
    println("Average magnitude of announce time changes: $(mean(stats["avg_change_magnitudes"]))")

    # Create summary evaluation plots
    summary_plots_dir = joinpath(eval_dir, "summary_plots")
    if !isdir(summary_plots_dir)
        mkpath(summary_plots_dir)
    end

    create_evaluation_plots(stats, summary_plots_dir)

    return stats
end

###
# Modified plotting code to work with MOMDPs
###

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

function plot_belief_distribution(belief, true_end_time, min_end_time, max_end_time, timestep, announced_time; title_prefix="")
    states, probs = extract_belief_states_and_probs(belief)
    
    # Extract only the Tt (true end time) component and its probability
    end_times = [s[2] for s in states]
    end_time_probs = Dict{Int, Float64}()
    
    # Aggregate probabilities by end time (may have multiple states with same end time)
    for (state, prob) in zip(states, probs)
        Tt = state[2]
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

function plot_observation_probability(pomdp, state, true_end_time, min_end_time, max_end_time, actual_observation=nothing; title_prefix="")
    t, Ta, Tt = state
    
    # Skip if at or past end time
    if t >= Tt
        return nothing
    end
    
    # Create dummy action to use in observation model
    a = Ta
    next_state = ((t+1, Ta), Tt)
    
    # Get observation distribution
    obs_dist = POMDPs.observation(pomdp, a, next_state)
    
    # If it's a deterministic distribution, convert to histogram format
    if obs_dist isa Deterministic
        o = obs_dist.val
        obs_time = o
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
        obs_times = [o for o in obs_list]
        
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

function plot_2d_belief_evolution(belief_history, true_end_time, min_end_time, max_end_time; title_prefix="")
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
            Tt = state[2]  # Extract true end time from state tuple (t, Ta, Tt)
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

function plot_2d_belief_evolution_with_actions(belief_history, run_details, true_end_time, min_end_time, max_end_time; title_prefix="")
    """
    Enhanced version that also overlays the announced times as a trajectory.
    """
    
    # Create the base 2D belief evolution plot
    p = plot_2d_belief_evolution(belief_history, true_end_time, min_end_time, max_end_time, title_prefix=title_prefix)
    
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
                announced_times[i]
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
                step["To"]
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
        max_end_time
    )
    if p_belief_2d !== nothing
        savefig(p_belief_2d, joinpath(plots_dir, "belief_evolution_2d.png"))
    end
    
    p_belief_2d_enhanced = plot_2d_belief_evolution_with_actions(
        belief_history,
        run_details,
        true_end_time, 
        min_end_time, 
        max_end_time
    )
    if p_belief_2d_enhanced !== nothing
        savefig(p_belief_2d_enhanced, joinpath(plots_dir, "belief_evolution_2d_with_actions.png"))
    end
    
    # Return the main plots
    return p_announce, p_reward
end

####
# Solve The Problem
####

MIN_END_TIME = 10
MAX_END_TIME = 20
DISCOUNT_FACTOR = 0.98

# For Evaluation
SEED = 0
NUM_SIMULATIONS = 10
VERBOSE = false
OUTPUT_DIR = "./results"
TRUE_END_TIME = nothing
INITIAL_ANNOUNCE = nothing


planning_momdp = PlanningProblem(MIN_END_TIME, MAX_END_TIME, DISCOUNT_FACTOR, nothing)

## Solve problem as MOMDP

solver_momdp = SARSOPSolver(; precision=0.5, timeout=300, pomdp_filename="planning_momdp.pomdpx", policy_filename="momdp_policy.policy", verbose=true)

println("Solving the MOMDP planning problem...")
ts = @elapsed begin
    # Solve the MOMDP using SARSOP
    policy_momdp = solve(solver_momdp, planning_momdp)

end
println("Solved the MOMDP in $(round(ts, digits=2)) seconds.")

evaluate_policy(
    planning_momdp, 
    policy_momdp, 
    NUM_SIMULATIONS, 
    OUTPUT_DIR,
    fixed_true_end_time=TRUE_END_TIME,
    initial_announce=INITIAL_ANNOUNCE,
    seed=SEED,
    verbose=VERBOSE,
    solver="MOMDP"
)

# # Save the policy to a file
# JLD2.@save "planning_momdp_policy.jld2" policy_momdp

## Solve problem by converting MOMDP POMDP

# planning_pomdp = MOMDPs.POMDP_of_Discrete_MOMDP(planning_momdp)
# solver_pomdp = SARSOPSolver(; precision=1e-2, timeout=180, pomdp_filename="planning_pomdp.pomdpx", policy_filename="pomdp_policy.policy", verbose=true)

# println("Solving the POMDP planning problem...")
# ts_pomdp = @elapsed begin
#     # Solve the POMDP using SARSOP
#     policy_pomdp = solve(solver_pomdp, planning_pomdp)
# end
# println("Solved the POMDP in $(round(ts_pomdp, digits=2)) seconds.")

# evaluate_policy(
#     planning_pomdp, 
#     policy_pomdp, 
#     NUM_SIMULATIONS, 
#     OUTPUT_DIR,
#     fixed_true_end_time=TRUE_END_TIME,
#     initial_announce=INITIAL_ANNOUNCE,
#     seed=SEED,
#     verbose=VERBOSE,
#     solver="MOMDP"
# )