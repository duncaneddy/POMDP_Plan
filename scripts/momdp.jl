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
using POMDPPlanning: create_debug_plots

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
MOMDPs.is_x_prime_dependent_on_y(::PlanningProblem) = false
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
    std = (Tt - t) / 1.5 # MAGIC NUMBER
    
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
        return 1.0
    end

    # Dead-simple reward - just penalize for the difference between announced and true end time
    return -1 * abs(Ta - Tt)
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

    updater = MOMDPDiscreteUpdater(momdp)

    for (b, s, a, o, r) in stepthrough(momdp, policy, updater, "b,s,a,o,r"; max_steps=1_000_000) # should be able to set max_steps=max_end_time+1
        r_sum += r
        step += 1
        t, Ta, Tt = s

        # Store belief for later plotting if requested
        if collect_beliefs
            push!(belief_history, deepcopy(b))
        end

        # Record first announced time
        if step == 1
            first_announced = Ta
        end

        # Track announcement changes
        if step > 1 && Ta != a.announced_time
            push!(announcement_changes, (t, Ta, a.announced_time, a.announced_time - Ta))
        end

        if verbose
            println("Timestep: ", t)
            println("True End Time: ", Tt)
            println("Previous Announced Time: ", Ta)
            println("Old Observation: ", obs_old)
            println("Action: ", a)
            println("Observed Time: ", o[3])
            obs_old = o[3]
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
            "action" => a.announced_time,
            "To" => o[3],
            "reward" => r,
            "cumulative_reward" => r_sum,
            "high_b_Tt" => highest_belief_state(b)[3],
            "belief_error" => abs(highest_belief_state(b)[3] - Tt),
            "announced_error" => abs(a.announced_time - Tt)
        )
        push!(iteration_details, iteration_detail)

        obs_old = o[3]
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
        # create_debug_plots(
        #     momdp,
        #     sim_metrics["iterations"],
        #     sim_metrics["min_end_time"],
        #     sim_metrics["max_end_time"],
        #     sim_dir,
        #     belief_history=sim_metrics["belief_history"],
        #     plot_beliefs=true,
        #     plot_observations=true
        # )

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

####
# Solve The Problem
####

MIN_END_TIME = 10
MAX_END_TIME = 20
DISCOUNT_FACTOR = 0.9999

planning_momdp = PlanningProblem(MIN_END_TIME, MAX_END_TIME, DISCOUNT_FACTOR, nothing)

## Solve problem as MOMDP

solver_momdp = SARSOPSolver(; precision=1e-2, timeout=180, pomdp_filename="planning_momdp.pomdpx", policy_filename="momdp_policy.policy", verbose=true)

println("Solving the MOMDP planning problem...")
ts = @elapsed begin
    # Solve the MOMDP using SARSOP
    policy_momdp = solve(solver_momdp, planning_momdp)

end
println("Solved the MOMDP in $(round(ts, digits=2)) seconds.")

# # Save the policy to a file
# JLD2.@save "planning_momdp_policy.jld2" policy_momdp

## Solve problem as POMDP

# planning_pomdp = MOMDPs.POMDP_of_Discrete_MOMDP(planning_momdp)
# solver_pomdp = SARSOPSolver(; precision=1e-2, timeout=180, pomdp_filename="planning_pomdp.pomdpx", policy_filename="pomdp_policy.policy", verbose=true)

# println("Solving the POMDP planning problem...")
# ts_pomdp = @elapsed begin
#     # Solve the POMDP using SARSOP
#     policy_pomdp = solve(solver_pomdp, planning_pomdp)
# end
# println("Solved the POMDP in $(round(ts_pomdp, digits=2)) seconds.")

# # Evaluate the policy
# SEED = 0
# NUM_SIMULATIONS = 10
# VERBOSE = false
# OUTPUT_DIR = "./results"
# TRUE_END_TIME = nothing
# INITIAL_ANNOUNCE = nothing



# evaluate_policy(
#     planning_momdp, 
#     policy_momdp, 
#     NUM_SIMULATIONS, 
#     OUTPUT_DIR,
#     fixed_true_end_time=TRUE_END_TIME,
#     initial_announce=INITIAL_ANNOUNCE,
#     seed=SEED,
#     verbose=VERBOSE,
#     solver="MOMDP"
# )