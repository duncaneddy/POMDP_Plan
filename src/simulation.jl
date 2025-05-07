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

function simulate_single_with_metrics(pomdp, policy; 
                                    fixed_true_end_time=nothing,
                                    initial_announce=nothing,
                                    verbose::Bool=false)
    step = 0
    r_sum = 0
    obs_old = NaN # dummy initialization
    iteration_details = []

    # Track metrics
    announcement_changes = []
    first_announced = nothing

    # Get domain parameters from POMDP
    state_list = states(pomdp)
    min_end_time = minimum([s[3] for s in state_list if s[1] == 0])
    max_end_time = maximum([s[3] for s in state_list if s[1] == 0])

    # Set up initial state with fixed true end time if specified
    if fixed_true_end_time !== nothing
        if fixed_true_end_time < min_end_time || fixed_true_end_time > max_end_time
            @warn "Fixed true end time $fixed_true_end_time outside valid range [$min_end_time, $max_end_time]"
            fixed_true_end_time = nothing
        end
    end

    # Set initial announced time if specified
    initial_Ta = initial_announce !== nothing ? initial_announce : min_end_time

    # Custom initialstate distribution if we have fixed parameters
    if fixed_true_end_time !== nothing || initial_announce !== nothing
        # If we have a fixed true end time, use it, otherwise sample from uniform
        if fixed_true_end_time !== nothing
            true_end_time = fixed_true_end_time
        else
            true_end_time = rand(min_end_time:max_end_time)
        end

        initial_state = (0, initial_Ta, true_end_time)
        initial_dist = Deterministic(initial_state)
    else
        initial_dist = initialstate(pomdp)
    end

    for (b, s, a, o, r) in stepthrough(pomdp, policy, DiscreteUpdater(pomdp), "b,s,a,o,r", initial_dist; max_steps=1_000_000)
        r_sum += r
        step += 1
        t, Ta, Tt = s

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
            print_belief_states_and_probs(b)
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
        "iterations" => iteration_details
    )

    return metrics
end

function simulate_many_with_metrics(pomdp, policy, num_simulations;
                                    fixed_true_end_time=nothing,
                                    initial_announce=nothing,
                                    verbose::Bool=false)
    println("Running $num_simulations evaluation simulation(s)")

    # Collect metrics across all simulations
    rewards = Float64[]
    initial_errors = Int[]
    final_errors = Int[]
    final_undershoot = Bool[]
    num_changes = Int[]
    avg_change_magnitudes = Float64[]
    std_change_magnitudes = Float64[]
    all_run_details = []

    progress = Progress(num_simulations, desc="Evaluating policy...")

    for i in 1:num_simulations
        if verbose && i % 10 == 0
            println("Simulation $i of $num_simulations")
        end

        # Run a single simulation and collect metrics
        metrics = simulate_single_with_metrics(
            pomdp, 
            policy,
            fixed_true_end_time=fixed_true_end_time,
            initial_announce=initial_announce,
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

        update!(progress, i)
    end

    # Compile statistics
    stats = Dict(
        "num_simulations" => num_simulations,
        "rewards" => rewards,
        "initial_errors" => initial_errors,
        "final_errors" => final_errors,
        "final_undershoot" => final_undershoot,
        "num_changes" => num_changes,
        "avg_change_magnitudes" => avg_change_magnitudes,
        "std_change_magnitudes" => std_change_magnitudes,
        "run_details" => all_run_details
    )

    return stats
end

# Add this to src/simulation.jl

function simulate_single_with_metrics_and_beliefs(pomdp, policy; 
                                                fixed_true_end_time=nothing,
                                                initial_announce=nothing,
                                                verbose::Bool=false)
    step = 0
    r_sum = 0
    obs_old = NaN # dummy initialization
    iteration_details = []
    belief_history = []

    # Track metrics
    announcement_changes = []
    first_announced = nothing

    # Get domain parameters from POMDP
    state_list = states(pomdp)
    min_end_time = minimum([s[3] for s in state_list if s[1] == 0])
    max_end_time = maximum([s[3] for s in state_list if s[1] == 0])

    # Set up initial state with fixed true end time if specified
    if fixed_true_end_time !== nothing
        if fixed_true_end_time < min_end_time || fixed_true_end_time > max_end_time
            @warn "Fixed true end time $fixed_true_end_time outside valid range [$min_end_time, $max_end_time]"
            fixed_true_end_time = nothing
        end
    end

    for (b, s, a, o, r) in stepthrough(pomdp, policy, DiscreteUpdater(pomdp), "b,s,a,o,r"; max_steps=1_000_000)
        r_sum += r
        step += 1
        t, Ta, Tt = s

        # Store belief for later plotting
        push!(belief_history, deepcopy(b))

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
            print_belief_states_and_probs(b)
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
        "belief_history" => belief_history
    )

    return metrics
end

function evaluate_policy(pomdp, policy, num_simulations, output_dir;
        fixed_true_end_time=nothing,
        initial_announce=nothing,
        seed=nothing,
        verbose::Bool=false,
        debug_plots::Bool=true)

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

    # Save evaluation parameters
    params = Dict(
        "num_simulations" => num_simulations,
        "fixed_true_end_time" => fixed_true_end_time,
        "initial_announce" => initial_announce,
        "seed" => seed,
        "timestamp" => string(Dates.now())
    )

    params_path = joinpath(output_dir, "evaluation_params.json")
    open(params_path, "w") do f
        JSON.print(f, params, 4)
    end

    # For more detailed debugging, run a single simulation with belief history
    if debug_plots
        println("Running simulation with belief history for debugging plots...")

        # Set seed for first simulation
        Random.seed!(seed)

        metrics_debug = simulate_single_with_metrics_and_beliefs(
            pomdp, 
            policy,
            fixed_true_end_time=fixed_true_end_time,
            initial_announce=initial_announce, 
            verbose=verbose
        )

        # Get domain parameters
        state_list = states(pomdp)
        min_end_time = minimum([s[3] for s in state_list if s[1] == 0])
        max_end_time = maximum([s[3] for s in state_list if s[1] == 0])

        # Create debug plots
        create_debug_plots(
            pomdp,
            metrics_debug["iterations"],
            min_end_time,
            max_end_time,
            output_dir,
            belief_history=metrics_debug["belief_history"]
        )
    end

    # Reset seed for main simulations
    Random.seed!(seed)

    # Run simulations with extended metrics
    stats = simulate_many_with_metrics(
        pomdp, 
        policy, 
        num_simulations,
        fixed_true_end_time=fixed_true_end_time,
        initial_announce=initial_announce, 
        verbose=verbose
    )

    # Add seed to stats
    stats["seed"] = seed

    # Save evaluation results
    results_filepath = joinpath(output_dir, "evaluation_results.json")
    open(results_filepath, "w") do f
        JSON.print(f, stats, 4)
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

    # Create standard evaluation plots
    create_evaluation_plots(stats, output_dir)

    return stats
end