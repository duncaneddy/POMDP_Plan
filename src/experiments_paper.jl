"""
Run comprehensive experiments for paper comparison across problem sizes and solvers.
Saves results incrementally to prevent memory growth.
"""
function run_paper_experiments(
    problem_sizes::Dict{String, Dict{String, Any}},
    solvers::Vector{String},
    output_dir::String;
    num_simulations::Int = 100,
    num_detailed_plots::Int = 10,  # Number of runs to save detailed belief evolution plots
    policy_timeout::Int = 300,
    discount_factor::Float64 = 0.98,
    seed::Union{Int, Nothing} = nothing,
    verbose::Bool = false,
    std_divisor::Float64 = 3.0,
    save_frequency::Int = 50  # Save results every N simulations
)
    # Set random seed
    if seed === nothing
        seed = rand(1:10000)
    end
    println("Using random seed: $seed")
    Random.seed!(seed)
    
    # Create experiment directory
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    experiment_dir = joinpath(output_dir, "paper_experiment_$(timestamp)")
    mkpath(experiment_dir)
    
    # Save experiment configuration
    config = Dict(
        "problem_sizes" => problem_sizes,
        "solvers" => solvers,
        "num_simulations" => num_simulations,
        "num_detailed_plots" => num_detailed_plots,
        "policy_timeout" => policy_timeout,
        "discount_factor" => discount_factor,
        "seed" => seed,
        "timestamp" => timestamp,
        "std_divisor" => std_divisor,
        "save_frequency" => save_frequency
    )
    
    config_path = joinpath(experiment_dir, "experiment_config.json")
    open(config_path, "w") do f
        JSON.print(f, config, 4)
    end
    
    println("Experiment configuration saved to: $config_path")
    
    # Results storage - only keep aggregated results in memory
    all_results = Dict{String, Dict{String, Any}}()
    
    # Run experiments for each problem size
    for (size_name, size_config) in problem_sizes
        println("\n" * "="^60)
        println("Running experiments for problem size: $size_name")
        println("="^60)
        
        # Load replay data for this problem size
        replay_data_path = size_config["replay_data_path"]
        println("Loading replay data from: $replay_data_path")
        replay_json = JSON.parsefile(replay_data_path)
        replay_data = replay_json["simulation_data"]
        
        # Ensure we have enough replay data
        if length(replay_data) < num_simulations
            error("Not enough replay data. Have $(length(replay_data)), need $num_simulations")
        end
        
        # Extract problem parameters
        min_end_time = size_config["min_end_time"]
        max_end_time = size_config["max_end_time"]
        
        # Create POMDP for this problem size
        pomdp = define_pomdp(
            min_end_time,
            max_end_time,
            discount_factor,
            verbose=verbose,
            std_divisor=std_divisor
        )
        
        # Also create MOMDP for MOMDP_SARSOP solver
        momdp = define_momdp(
            min_end_time,
            max_end_time,
            discount_factor,
            std_divisor=std_divisor
        )
        
        # Generate policies for each solver
        policies = Dict{String, Any}()
        policy_times = Dict{String, Float64}()
        
        for solver_type in solvers
            println("\nGenerating policy with $solver_type solver...")
            
            if uppercase(solver_type) == "MOMDP_SARSOP"
                policy_data = get_policy(momdp, solver_type, experiment_dir, 
                                       verbose=verbose, policy_timeout=policy_timeout)
            else
                policy_data = get_policy(pomdp, solver_type, experiment_dir, 
                                       verbose=verbose, policy_timeout=policy_timeout)
            end
            
            policies[solver_type] = policy_data["policy"]
            policy_times[solver_type] = policy_data["policy_solve_time"]
        end
        
        # Run simulations for each solver with incremental saving
        size_results = Dict{String, Any}()
        
        for solver_type in solvers
            println("\nRunning simulations with $solver_type solver...")
            
            # Use appropriate POMDP type
            problem = uppercase(solver_type) == "MOMDP_SARSOP" ? momdp : pomdp
            
            # Run simulations with incremental saving
            sim_results = simulate_many_incremental(
                deepcopy(problem),
                policies[solver_type],
                num_simulations,
                experiment_dir,
                size_name,
                solver_type,
                replay_data[1:num_simulations],
                num_detailed_plots,
                save_frequency,
                seed,
                verbose
            )
            
            sim_results["policy_solve_time"] = policy_times[solver_type]
            
            # Store consolidated results
            size_results[solver_type] = sim_results
        end
        
        # Save size results (consolidated)
        size_results_path = joinpath(experiment_dir, "results_$(size_name).json")
        save_consolidated_results(size_results, size_results_path)
        
        all_results[size_name] = size_results
        
        # Generate detailed belief evolution plots for a subset of runs
        if num_detailed_plots > 0
            generate_detailed_belief_plots_from_files(
                experiment_dir,
                size_name,
                solvers,
                pomdp,
                momdp,
                num_detailed_plots
            )
        end
        
        # Force garbage collection after each problem size
        GC.gc()
    end
    
    # Save combined results (consolidated only)
    combined_results_path = joinpath(experiment_dir, "all_results.json")
    save_consolidated_results(all_results, combined_results_path)
    
    println("\n" * "="^60)
    println("Experiment complete!")
    println("Results saved to: $experiment_dir")
    println("="^60)
    
    return experiment_dir, all_results
end

"""
Run simulations with incremental saving to prevent memory growth.
"""
function simulate_many_incremental(
    pomdp, 
    policy, 
    num_simulations::Int,
    experiment_dir::String,
    size_name::String,
    solver_type::String,
    replay_data::Vector,
    num_detailed_plots::Int,
    save_frequency::Int,
    seed::Int,
    verbose::Bool
)
    # Set random seed
    Random.seed!(seed)
    rand_range = 1:100_000_000
    
    # Create directories for detailed data
    detailed_dir = joinpath(experiment_dir, "detailed_data", size_name, solver_type)
    mkpath(detailed_dir)
    
    # Initialize consolidated metrics
    consolidated_metrics = Dict{String, Any}(
        "rewards" => Float64[],
        "initial_errors" => Int[],
        "final_errors" => Int[],
        "num_changes" => Int[],
        "avg_change_magnitudes" => Float64[],
        "std_change_magnitudes" => Float64[],
        "final_undershoot" => Bool[]
    )
    
    # Track simulation data for reproduction
    # all_simulation_data = []
    
    # Temporary storage for batch saving
    batch_metrics = []
    batch_detailed = []
    
    println("Running $num_simulations simulation(s) with incremental saving every $save_frequency simulations")
    progress = Progress(num_simulations, desc="Running simulations...")
    
    for i in 1:num_simulations
        # Sample a random integer seed for this run
        run_seed = rand(rand_range)
        
        # Determine if we need detailed data for this simulation
        collect_detailed = i <= num_detailed_plots
        
        # Run a single simulation
        metrics = simulate_single(
            pomdp, 
            policy,
            collect_beliefs=collect_detailed,
            verbose=false,  # Disable verbose for individual sims
            debug=false,
            replay_data=replay_data[i],
            seed=run_seed
        )
        
        if metrics === nothing
            if verbose
                println("Simulation $i failed, skipping...")
            end
            continue
        end
        
        # Store consolidated metrics
        push!(consolidated_metrics["rewards"], metrics["total_reward"])
        push!(consolidated_metrics["initial_errors"], metrics["initial_error"])
        push!(consolidated_metrics["final_errors"], metrics["final_error"])
        push!(consolidated_metrics["num_changes"], metrics["num_changes"])
        push!(consolidated_metrics["avg_change_magnitudes"], metrics["avg_change_magnitude"])
        push!(consolidated_metrics["std_change_magnitudes"], metrics["std_change_magnitude"])
        push!(consolidated_metrics["final_undershoot"], metrics["final_undershoot"])
        
        # Store simulation data for reproduction
        # push!(all_simulation_data, metrics["simulation_data"])
        
        # Add to batch for detailed saving
        if collect_detailed
            # Create minimal detailed metrics for this simulation
            detailed_metrics = Dict(
                "simulation_id" => i,
                "total_reward" => metrics["total_reward"],
                "initial_error" => metrics["initial_error"],
                "final_error" => metrics["final_error"],
                "num_changes" => metrics["num_changes"],
                "iterations" => metrics["iterations"],
                "belief_history" => metrics["belief_history"],
                "min_end_time" => metrics["min_end_time"],
                "max_end_time" => metrics["max_end_time"]
            )
            push!(batch_detailed, detailed_metrics)
        end
        
        # Save batches periodically
        if i % save_frequency == 0 || i == num_simulations
            # Save consolidated metrics batch
            batch_file = joinpath(detailed_dir, "consolidated_batch_$(div(i-1, save_frequency) + 1).json")
            batch_data = Dict(
                "batch_start" => max(1, i - save_frequency + 1),
                "batch_end" => i,
                "metrics" => consolidated_metrics
            )
            save_json_safe(batch_data, batch_file)
            
            # Save detailed data batch if any
            if !isempty(batch_detailed)
                detailed_batch_file = joinpath(detailed_dir, "detailed_batch_$(div(i-1, save_frequency) + 1).json")
                save_json_safe(batch_detailed, detailed_batch_file)
                batch_detailed = []  # Clear batch
            end
            
            # if verbose
            #     println("Saved batch ending at simulation $i")
            # end
        end
        
        update!(progress, i)
    end
    
    # Save final simulation data for reproduction
    # sim_data_file = joinpath(detailed_dir, "simulation_data.json")
    # save_json_safe(Dict("simulation_data" => all_simulation_data), sim_data_file)
    
    # Return consolidated results
    return consolidated_metrics
end

"""
Recursively clean data structure to remove non-JSON-serializable objects.
"""
function clean_for_json(obj)
    if obj isa Dict
        cleaned = Dict()
        for (key, value) in obj
            # Handle belief_history specially
            if key == "belief_history"
                cleaned_value = serialize_belief_history(value)
                if cleaned_value !== nothing
                    cleaned[string(key)] = cleaned_value
                end
                continue
            end
            
            # Skip other known problematic keys
            if key in ["policy", "pomdp", "momdp", "belief", 
                      "updater", "rng", "policy_object"]
                continue
            end
            
            cleaned_value = clean_for_json(value)
            if cleaned_value !== nothing
                cleaned[string(key)] = cleaned_value
            end
        end
        return cleaned
        
    elseif obj isa Array || obj isa Vector
        cleaned = []
        for item in obj
            cleaned_item = clean_for_json(item)
            if cleaned_item !== nothing
                push!(cleaned, cleaned_item)
            end
        end
        return cleaned
        
    elseif obj isa Tuple
        # Convert tuples to arrays for JSON compatibility
        return clean_for_json(collect(obj))
        
    elseif obj isa Number || obj isa String || obj isa Bool || obj === nothing
        return obj
        
    elseif obj isa Symbol
        return string(obj)
        
    else
        # Skip non-serializable objects (functions, policies, etc.)
        return nothing
    end
end


"""
Convert belief history to JSON-serializable format.
"""
function serialize_belief_history(belief_history)
    if belief_history === nothing
        return nothing
    end
    
    serialized_beliefs = []
    
    for belief in belief_history
        if belief === nothing
            push!(serialized_beliefs, nothing)
            continue
        end
        
        try
            states, probs = extract_belief_states_and_probs(belief)
            
            # Convert states to arrays for JSON compatibility
            serialized_states = []
            for state in states
                if isa(state, Tuple)
                    push!(serialized_states, collect(state))
                else
                    push!(serialized_states, state)
                end
            end
            
            serialized_belief = Dict(
                "states" => serialized_states,
                "probabilities" => collect(probs),
                "belief_type" => string(typeof(belief))
            )
            
            push!(serialized_beliefs, serialized_belief)
        catch e
            # If we can't serialize this belief, store a placeholder
            push!(serialized_beliefs, Dict(
                "error" => "Failed to serialize belief: $(e)",
                "belief_type" => string(typeof(belief))
            ))
        end
    end
    
    return serialized_beliefs
end

"""
Reconstruct belief objects from serialized format for plotting.
"""
function deserialize_belief_history(serialized_beliefs, is_momdp::Bool=false)
    if serialized_beliefs === nothing
        return nothing
    end
    
    beliefs = []
    
    for serialized_belief in serialized_beliefs
        if serialized_belief === nothing
            push!(beliefs, nothing)
            continue
        end
        
        if haskey(serialized_belief, "error")
            # Skip beliefs that failed to serialize
            push!(beliefs, nothing)
            continue
        end
        
        try
            states = serialized_belief["states"]
            probs = serialized_belief["probabilities"]
            
            # Convert states back to tuples if needed
            converted_states = []
            for state in states
                if isa(state, Array)
                    if is_momdp && length(state) == 1
                        # For MOMDP, states are just integers
                        push!(converted_states, state[1])
                    else
                        # For POMDP, states are tuples
                        push!(converted_states, Tuple(state))
                    end
                else
                    push!(converted_states, state)
                end
            end
            
            # Create a simple belief representation that works with existing functions
            belief = POMDPTools.POMDPDistributions.SparseCat(converted_states, probs)
            push!(beliefs, belief)
        catch e
            # If reconstruction fails, use a placeholder
            push!(beliefs, nothing)
        end
    end
    
    return beliefs
end

"""
Safe JSON saving with memory cleanup.
"""
function save_json_safe(data, filepath::String)
    # Clean data for JSON serialization
    cleaned_data = clean_for_json(data)
    
    open(filepath, "w") do f
        JSON.print(f, cleaned_data, 4)
    end
    
    # Force cleanup
    cleaned_data = nothing
    GC.gc()
end

"""
Save only consolidated results to prevent memory issues.
"""
function save_consolidated_results(results::Dict, filepath::String)
    # Only save the essential consolidated metrics
    consolidated = Dict()
    
    for (key, value) in results
        if isa(value, Dict)
            consolidated[key] = Dict()
            for (subkey, subvalue) in value
                if isa(subvalue, Dict) && haskey(subvalue, "rewards")
                    # This is a solver result - keep only essential metrics
                    consolidated[key][subkey] = Dict(
                        "rewards" => subvalue["rewards"],
                        "initial_errors" => subvalue["initial_errors"],
                        "final_errors" => subvalue["final_errors"],
                        "num_changes" => subvalue["num_changes"],
                        "avg_change_magnitudes" => subvalue["avg_change_magnitudes"],
                        "std_change_magnitudes" => subvalue["std_change_magnitudes"],
                        "final_undershoot" => subvalue["final_undershoot"],
                        "policy_solve_time" => get(subvalue, "policy_solve_time", 0.0)
                    )
                else
                    consolidated[key][subkey] = subvalue
                end
            end
        else
            consolidated[key] = value
        end
    end
    
    save_json_safe(consolidated, filepath)
end

"""
Generate detailed belief evolution plots from saved files instead of memory.
"""
function generate_detailed_belief_plots_from_files(
    experiment_dir::String,
    size_name::String,
    solvers::Vector{String},
    pomdp,
    momdp,
    num_detailed_plots::Int
)
    plots_dir = joinpath(experiment_dir, "belief_evolution_plots", size_name)
    mkpath(plots_dir)
    
    for solver in solvers
        solver_dir = joinpath(plots_dir, solver)
        mkpath(solver_dir)
        
        # Use appropriate problem formulation
        problem = uppercase(solver) == "MOMDP_SARSOP" ? momdp : pomdp
        is_momdp = isa(problem, PlanningProblem)
        
        # Load detailed data from files
        detailed_dir = joinpath(experiment_dir, "detailed_data", size_name, solver)
        
        if !isdir(detailed_dir)
            continue
        end
        
        # Find all detailed batch files
        batch_files = filter(f -> startswith(f, "detailed_batch"), readdir(detailed_dir))
        
        plot_count = 0
        for batch_file in batch_files
            if plot_count >= num_detailed_plots
                break
            end
            
            batch_path = joinpath(detailed_dir, batch_file)
            try
                batch_data = JSON.parsefile(batch_path)
                
                for detailed_metrics in batch_data
                    if plot_count >= num_detailed_plots
                        break
                    end
                    
                    plot_count += 1
                    
                    # Extract and deserialize belief history
                    serialized_belief_history = get(detailed_metrics, "belief_history", nothing)
                    belief_history = deserialize_belief_history(serialized_belief_history, is_momdp)
                    run_details = detailed_metrics["iterations"]
                    
                    if belief_history === nothing || isempty(belief_history)
                        println("Warning: No belief history available for $(solver) run $(detailed_metrics["simulation_id"])")
                        continue
                    end
                    
                    true_end_time = run_details[1]["Tt"]
                    
                    # Create 2D belief evolution plot with actions
                    p = plot_2d_belief_evolution_with_actions(
                        belief_history,
                        run_details,
                        true_end_time,
                        detailed_metrics["min_end_time"],
                        detailed_metrics["max_end_time"],
                        title_prefix="$solver - Run $(detailed_metrics["simulation_id"]) - ",
                        is_momdp=is_momdp
                    )
                    
                    if p !== nothing
                        savefig(p, joinpath(solver_dir, "belief_evolution_run_$(lpad(detailed_metrics["simulation_id"], 3, '0')).png"))
                    end
                end
            catch e
                println("Warning: Could not load detailed batch file $batch_file: $e")
                continue
            end
        end
    end
end