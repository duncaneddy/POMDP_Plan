using Pkg
using Dates

using POMDPs
using QuickPOMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPModelTools
using QMDP
using LinearAlgebra
using Random
using NativeSARSOP
using Distributions
using POMDPTools
using FIB
using PyPlot
using Debugger
using ProgressMeter

# Parameters
global max_end_time = 10
global max_estimated_time = 12
global min_end_time = 6
global min_estimated_time = 5
global discount_factor = 0.95

# Define a structured action type for announcing a specific time
struct AnnounceAction
    announced_time::Int
end

function define_pomdp()
    # Define actions: one "dont_announce" plus an AnnounceAction for each possible time
    base_actions = [AnnounceAction(To_val) for To_val in min_estimated_time:max_estimated_time]
    actions = vcat([:dont_announce], base_actions)

    pomdp = QuickPOMDP(
        states = [(t, Ta, Ts) for t in 0:max_end_time,
                                 Ta in min_estimated_time:max_estimated_time,
                                 Ts in min_end_time:max_end_time],

        actions = actions,
        actiontype = Union{Symbol, AnnounceAction},
        observations = [(t, Ta, To) for t in 0:max_end_time,
                                     Ta in min_estimated_time:max_estimated_time,
                                     To in min_estimated_time:max_estimated_time],

        discount = discount_factor,

        transition = function(s, a)
            t, Ta, Ts = s
            # Move time forward, but not beyond the true end time
            t = min(t + 1, Ts)

            if a == :dont_announce
                new_state = (t, Ta, Ts)
            elseif a isa AnnounceAction
                # Update Ta to the announced_time chosen by the action
                new_Ta = a.announced_time
                new_state = (t, new_Ta, Ts)
            else
                error("Invalid action: $a")
            end

            return Deterministic(new_state)
        end,

        observation = function(a, sp)
            t, Ta, Ts = sp
            possible_Tos = collect(min_estimated_time:max_estimated_time)

            # If the project is done or no variance scenario:
            # just return Ts deterministically
            if t >= Ts - 1
                return Deterministic((t, Ta, Ts))
            end

            # Calculate parameters of the truncated normal
            mu = Ts
            std = (Ts - t) / 3
            if std <= 0
                # If no variance, fall back to Ts deterministically
                return Deterministic((t, Ta, Ts))
            end

            base_dist = Normal(mu, std)
            lower = min_estimated_time
            upper = max_estimated_time
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
                return Deterministic((t, Ta, Ts))
            end
            probs ./= total_p

            obs_list = [(t, Ta, To_val) for To_val in possible_Tos]
            return SparseCat(obs_list, probs)
        end,

        reward = function(s, a)
            t, Ta, Ts = s
            earlier = -8
            later = -10

            # If announcing an impossible time
            if (a != :dont_announce && a.announced_time < t) || (Ts == t && Ta != t)
                return -1000000000.0
            end

            r = -1 * abs(Ta - Ts)
            

            if a != :dont_announce
                time_to_end = Ts - t
                if Ta < Ts
                    r += earlier * (1 / time_to_end)
                elseif Ta > Ts
                    r += later * (1 / time_to_end)
                end
            end
            return r
        end,

        initialstate = function()
            possible_states = [(0, Ta, Ts) for Ta in min_estimated_time:max_estimated_time,
                                            Ts in min_end_time:max_end_time]
            num_states = length(possible_states)
            probabilities = fill(1.0 / num_states, num_states)
            return SparseCat(possible_states, probabilities)
        end
    )
    return pomdp
end

function print_belief_states_and_probs(belief)
    states = belief.state_list
    probs = belief.b

    println("States and their probabilities:")
    for (state, prob) in zip(states, probs)
        if prob > 0
            println("State: $state, Probability: $prob")
        end
    end
end

function get_policy(pomdp, solver_type)
    println("Computing policy")
    if solver_type == "random"
        elapsed_time = @elapsed policy = RandomPolicy(pomdp)
    elseif solver_type == "fib"
        elapsed_time = @elapsed policy = solve(FIBSolver(), pomdp)
    elseif solver_type == "qmdp"
        elapsed_time = @elapsed policy = solve(QMDPSolver(), pomdp)
    elseif solver_type == "sarsop"
        elapsed_time = @elapsed policy = solve(SARSOPSolver(), pomdp)
    else
        println("Invalid solver type: $solver_type. Using random policy by default.")
        elapsed_time = @elapsed policy = RandomPolicy(pomdp)
    end
    println("Time to compute policy: ", elapsed_time, " seconds")
    return policy
end

function simulate_single(pomdp, policy; verbose=true)
    step = 0
    r_sum = 0

    for (b, s, a, o, r) in stepthrough(pomdp, policy, DiscreteUpdater(pomdp), "b,s,a,o,r"; max_steps=1_000_000)
        r_sum += r
        step += 1
        t, Ta, Ts = s
        if verbose
            println("Step $step")
            println("Action: ", a)
            println("Timestep: $t, Announced Time: $Ta, True End Time: $Ts")
            println("Observation: $o")
            print_belief_states_and_probs(b)
            @show r r_sum
            println()
        end

        if t == Ts
            if verbose
                println("Project complete!")
            end
            break
        end
    end
    return r_sum
end

function simulate_many(pomdp, solver_type, num_simulations)
    println("Simulating $num_simulations times")
    policy = get_policy(pomdp, solver_type)
    println("Computed Policy")
    total_reward = 0
    rewards = []
    progress = Progress(num_simulations, desc="Running simulations")
    iteration_times = []

    for i in 1:num_simulations
        if i % 100 == 0
            println("Simulation: ", i, " ")
        end
        start_time = now()
        r = simulate_single(pomdp, policy, verbose=false)
        end_time = now()

        total_reward += r
        push!(rewards, r)
        push!(iteration_times, end_time - start_time)
        update!(progress, i)
    end
    println("Total reward: ", total_reward)
    println("Average reward: ", total_reward / num_simulations)
    numeric_times = [time.value for time in iteration_times]
    average_time = mean(numeric_times)
    println("Average iteration time in simulation: ", average_time / 1000, " seconds")

    return rewards
end

function simulate(pomdp, solver_type)
    println("Using solver: $solver_type")
    policy = get_policy(pomdp, solver_type)
    simulate_single(pomdp, policy)
end 

function plot_rewards(rewards, solver_type)
    hist(rewards, bins=30)
    xlabel("Reward")
    ylabel("Frequency")
    savefig("plots/histogram_$(solver_type).png")
end

function main()
    if length(ARGS) < 1
        println("Usage: julia script_name.jl <solver_type>")
        println("Available solvers: random, fib, qmdp, sarsop")
        return
    end
    
    solver_type = ARGS[1]
    pomdp = define_pomdp()
    simulate(pomdp, solver_type)
    rewards = simulate_many(pomdp, solver_type, 100)
    plot_rewards(rewards, solver_type)
end

main()
