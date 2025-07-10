mutable struct PlanningProblem <: MOMDP{Tuple{Int, Int}, Int, Int, Int, Float64}
    min_end_time::Int
    max_end_time::Int
    discount_factor::Float64
    initial_announced_time::Union{Int, Nothing}
    std_divisor::Float64
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

    return create_momdp_observation(t, Tt, problem.min_end_time, problem.max_end_time, std_divisor=problem.std_divisor)

end

function POMDPs.reward(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int}, action::Int)
    t, Ta = state[1]
    Tt = state[2]

    return calculate_reward(t, Ta, Tt, action, problem.min_end_time, problem.max_end_time)
end

function POMDPs.isterminal(problem::PlanningProblem, state::Tuple{Tuple{Int, Int}, Int})
    # The project is done if the current time t is greater than or equal to the true end time Tt
    return state[1][1] >= state[2]
end

function define_momdp(
    min_end_time::Int=10, 
    max_end_time::Int=20, 
    discount_factor::Float64=0.975;
    initial_announce::Union{Int, Nothing}=nothing,
    std_divisor::Float64=3.0
)
    return PlanningProblem(
        min_end_time,
        max_end_time,
        discount_factor,
        initial_announce,
        std_divisor
    )
end