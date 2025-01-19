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
using JSON

# Parameters
global max_end_time = 10
global max_estimated_time = 12
global min_end_time = 6
global min_estimated_time = 5
global discount_factor = 0.95
global NUM_SIMULATIONS = 1000
global WRONG_END_TIME_REWARD = -100

include("mostlikely.jl")

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
                # Note that the action can be any number in min_estimated_time:max_estimated_time 
                # The paper restricts this to only the previous observed time
                new_Ta = a.announced_time
                new_state = (t, new_Ta, Ts)
            else
                error("Invalid action: $a")
            end

            return Deterministic(new_state)
        end,

        observation = function(a, sp)
            t, Ta, Ts = sp
            min_obs_time = max(t + 1, min_estimated_time)
            # possible observed times must be after the current timestep
            possible_Tos = collect(min_obs_time:max_estimated_time)

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
            # the truncated normal distribution is defined from t+1 to max_estimated_time
            lower = min_obs_time
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
                return WRONG_END_TIME_REWARD
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
    elseif solver_type == "mostlikely"
        elapsed_time = @elapsed policy = MostLikelyPolicy()
    else
        println("Invalid solver type: $solver_type. Using random policy by default.")
        elapsed_time = @elapsed policy = RandomPolicy(pomdp)
    end
    println("Time to compute policy: ", elapsed_time, " seconds")
    output = Dict(
        "policy" => policy,
        "comp_policy_time" => elapsed_time
    )
    return output
end

function simulate_single(pomdp, policy; verbose=true)
    step = 0
    r_sum = 0
    obs_old = NaN # dummy initialization
    iteration_details = [] 
    for (b, s, a, o, r) in stepthrough(pomdp, policy, DiscreteUpdater(pomdp), "b,s,a,o,r"; max_steps=1_000_000)
        r_sum += r
        step += 1
        t, Ta, Ts = s
        if verbose || r == WRONG_END_TIME_REWARD
            println("Timestep: ", t)
            println("True End Time: ", Ts)
            println("Previous Announced Time: ", Ta)
            println("Old Observation: ", obs_old)
            println("Action: ", a)
            println("Observed Time: ", o[3])
            obs_old = o[3]
            print_belief_states_and_probs(b)
            @show r r_sum
            println()
        end
        
        # save stats for analysis
        iteration_detail = Dict(
            "timestep" => t,
            "Ts" => Ts,
            "Ta_prev" => Ta,
            "To_prev" => obs_old,
            "action" => a,
            "To" => o[3],
            "reward" => r
        )
        push!(iteration_details, iteration_detail)

        obs_old = o[3]
        if t == Ts
            if verbose
                println("Project complete!")
            end
            break
        end
    end
    return iteration_details
end

function simulate_many(pomdp, solver_type, num_simulations)
    println("Simulating $num_simulations times")
    policy_out = get_policy(pomdp, solver_type)
    policy = policy_out["policy"]
    println("Computed Policy")
    total_reward = 0
    rewards = []
    run_details = Vector{Vector{Dict}}(undef, 0)

    progress = Progress(num_simulations, desc="Running simulations")
    iteration_times = []

    for i in 1:num_simulations
        if i % 100 == 0
            println("Simulation: ", i, " ")
        end
        start_time = now()
        stats_sing = simulate_single(pomdp, policy, verbose=false)
        r = sum([step["reward"] for step in stats_sing])
        # add stats for this run to the run_details
        push!(run_details, stats_sing)

        end_time = now()

        total_reward += r
        push!(rewards, r)
        push!(iteration_times, end_time - start_time)
        update!(progress, i)
    end
    println("Total reward: ", total_reward)
    println("Average reward: ", total_reward / num_simulations)
    iter_times = [time.value / 1000 for time in iteration_times] # 1000 to convert to seconds

    average_time = mean(iter_times)
    println("Average iteration time in simulation: ", average_time, " seconds")
    
    stats = Dict(
        "solver_type" => solver_type,
        "comp_policy_time" => policy_out["comp_policy_time"],
        "rewards" => rewards, 
        "iteration_times" => iter_times, 
        "Ts_min" => min_end_time,
        "Ts_max" => max_end_time,
        "To_min" => min_estimated_time,
        "To_max" => max_estimated_time,
        "run_details" => run_details
    )
    return stats
end

function simulate(pomdp, solver_type)
    println("Using solver: $solver_type")
    policy_out = get_policy(pomdp, solver_type)
    policy = policy_out["policy"]
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
        println("Available solvers: random, fib, qmdp, sarsop, mostlikely")
        return
    end
    
    solver_type = ARGS[1]
    pomdp = define_pomdp()
    simulate(pomdp, solver_type)
    # results is a dictionary with keys rewards, numeric_times
    results = simulate_many(pomdp, solver_type, NUM_SIMULATIONS) 
    write("results/$(solver_type)_results.json", JSON.json(results))
    plot_rewards(results["rewards"], solver_type)
end

main()
