function simulate_single(pomdp, policy; verbose::Bool=true)
    step = 0
    r_sum = 0
    obs_old = NaN # dummy initialization
    iteration_details = [] 
    for (b, s, a, o, r) in stepthrough(pomdp, policy, DiscreteUpdater(pomdp), "b,s,a,o,r"; max_steps=1_000_000)
        r_sum += r
        step += 1
        t, Ta, Tt = s
        if verbose || r == WRONG_END_TIME_REWARD
            println("Timestep: ", t)
            println("True End Time: ", Tt)
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
            "Tt" => Tt,
            "Ta_prev" => Ta,
            "To_prev" => obs_old,
            "action" => a,
            "To" => o[3],
            "reward" => r,
            "high_b_Tt" => highest_belief_state(b)[3], 
            "t" => t,
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
    return iteration_details
end

function simulate_many(pomdp, solver_type, num_simulations, policy; verbose::Bool=true)
    println("Simulating $num_simulations times")
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
        "rewards" => rewards, 
        "iteration_times" => iter_times, 
        "Tt_min" => min_end_time,
        "Tt_max" => max_end_time,
        "To_min" => min_end_time,
        "To_max" => max_end_time,
        "run_details" => run_details
    )
    return stats
end