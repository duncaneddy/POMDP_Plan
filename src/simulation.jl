function simulate_single(pomdp, policy; 
                        fixed_true_end_time=nothing,
                        initial_announce=nothing,
                        collect_beliefs=true,
                        verbose=false)
    is_momdp = isa(pomdp, PlanningProblem)
    println("Running simulation with $(is_momdp ? "MOMDP" : "POMDP") formulation")
    step = 0
    r_sum = 0
    obs_old = NaN # dummy initialization
    iteration_details = []
    belief_history = collect_beliefs ? [] : nothing

    # Track metrics
    announcement_changes = []
    first_announced = nothing

    # Get domain parameters from POMDP
    if is_momdp
        true_end_times = states_y(pomdp)
        min_end_time = minimum(true_end_times)
        max_end_time = maximum(true_end_times)
    else
        state_list = states(pomdp)
        min_end_time = minimum([s[3] for s in state_list if s[1] == 0])
        max_end_time = maximum([s[3] for s in state_list if s[1] == 0])
    end
    

    # Set up initial state with fixed true end time if specified
    if fixed_true_end_time !== nothing
        if fixed_true_end_time < min_end_time || fixed_true_end_time > max_end_time
            @warn "Fixed true end time $fixed_true_end_time outside valid range [$min_end_time, $max_end_time]"
            fixed_true_end_time = nothing
        end
    end

    if typeof(policy) == ObservedTimePolicy
        # Use PreviousObservationUpdater for ObservedTimePolicy
        updater = PreviousObservationUpdater()
    else
        updater = DiscreteUpdater(pomdp)
    end


    for (b, s, a, o, r) in stepthrough(pomdp, policy, updater, "b,s,a,o,r"; max_steps=1_000_000) # should be able to set max_steps=max_end_time+1
        r_sum += r
        step += 1

        if is_momdp
            t, Ta = s[1]
            Tt = s[2]
        else
            t, Ta, Tt = s
        end

        # Store belief for later plotting if requested
        if collect_beliefs
            push!(belief_history, deepcopy(b))
        end

        # Record first announced time
        if step == 1
            first_announced = Ta
        end

        # Track announcement changes
        if is_momdp
            a_value = a
            obs = o
        else
            a_value = a.announced_time  # Extract action value from AnnounceAction
            obs = o[3]
        end

        if step > 1 && Ta != a
            push!(announcement_changes, (t, Ta, a_value, a_value - Ta))
        end

        if verbose
            println("Timestep: ", t)
            println("True End Time: ", Tt)
            println("Previous Announced Time: ", Ta)
            println("Old Observation: ", obs_old)
            println("Action: ", a_value)
            println("Observed Time: ", obs)
            obs_old = obs
            print_belief_states_and_probs(b)
            @show r r_sum
            println()
        end

        if is_momdp
            believed_Tt = highest_belief_state(b)[2]
        else
            believed_Tt = highest_belief_state(b)[3]
        end
        

        # Save detailed metrics for this step
        iteration_detail = Dict(
            "timestep" => t,
            "Tt" => Tt,
            "Ta_prev" => Ta,
            "To_prev" => obs_old,
            "action" => a_value,
            "To" => obs,
            "reward" => r,
            "cumulative_reward" => r_sum,
            "high_b_Tt" => believed_Tt,
            "belief_error" => abs(believed_Tt - Tt),
            "announced_error" => abs(a_value - Tt)
        )
        push!(iteration_details, iteration_detail)

        obs_old = obs
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
function simulate_many(pomdp, policy, num_simulations;
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
            pomdp, 
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

function evaluate_policy(pomdp, policy, num_simulations, output_dir;
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
        pomdp, 
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
            pomdp,
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