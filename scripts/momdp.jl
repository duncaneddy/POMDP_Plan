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

function MOMDPs.initialstate_x(problem::PlanningProblem)
    if problem.initial_announced_time !== nothing
        return (0, problem.initial_announced_time)
    else
        return (0, problem.min_end_time)  # Default to min_end_time if not specified
    end
end

function MOMDPs.initialstate_y(problem::PlanningProblem)
    num_states_y = length(states_y(problem))
    return SparseCat(states_y(problem), fill(1.0 / num_states_y, num_states_y))
end

function MOMDPs.transition_x(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int)
    return Deterministic((state[1][1] + 1, action))
end

function MOMDPs.transition_y(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int)
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

function POMDPs.observation(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int)
    t, Ta = state[1]
    Tt = state[2]

    # If the project is done return the true end time Tt
    if t >= Tt || t + 1 == problem.max_end_time
        return Deterministic(Tt)
    end

    min_obs_time = max(t + 1, min_end_time)

    possible_Tos = collect(min_obs_time:max_end_time)

    # Calculate parameters of the truncated normal
    mu = Tt
    std = (Tt - t) / 1.5 # MAGIC NUMBER
    
    if Tt - t <= 0 
        # The task is done, so we should observe the true end time
        return Deterministic(Tt)
    end

    base_dist = Normal(mu, std)

    println("Base distribution: $base_dist, mu: $mu, std: $std")

    # the truncated normal distribution is defined from t+1 to max_end_time
    lower = min_obs_time
    upper = max_end_time
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
# Solve The Problem
####

MIN_END_TIME = 10
MAX_END_TIME = 20
DISCOUNT_FACTOR = 0.9999

planning_momdp = PlanningProblem(MIN_END_TIME, MAX_END_TIME, DISCOUNT_FACTOR, nothing)

solver_momdp = SARSOPSolver(; precision=1e-2, timeout=180, pomdp_filename="planning_momdp.pomdpx", verbose=false)
policy_momdp = solve(solver_momdp, planning_momdp)