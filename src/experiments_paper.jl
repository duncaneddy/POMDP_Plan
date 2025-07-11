"""
Run comprehensive experiments for paper comparison across problem sizes and solvers.
Saves all raw data for later processing.
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
    std_divisor::Float64 = 3.0
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
        "std_divisor" => std_divisor
    )
    
    config_path = joinpath(experiment_dir, "experiment_config.json")
    open(config_path, "w") do f
        JSON.print(f, config, 4)
    end
    
    println("Experiment configuration saved to: $config_path")
    
    # Results storage
    all_results = Dict{String, Dict{String, Any}}()  # problem_size => solver => results
    
    # Run experiments for each problem size
    for (size_name, size_config) in problem_sizes
        println("\n" * "="^60)
        println("Running experiments for problem size: $size_name")
        println("="^60)
        
        size_results = Dict{String, Any}()
        
        # Load replay data for this problem size
        replay_data_path = size_config["replay_data_path"]
        println("Loading replay data from: $replay_data_path")
        replay_json = JSON.parsefile(replay_data_path)
        replay_data = replay_json["simulation_data"]
        
        # Ensure we have enough replay data
        if length(replay_data) < num_simulations
            error("Not enough replay data. Have $(length(replay_data)), need $num_simulations")
        end
        
        # Extract problem parameters from first simulation
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
        
        # Run simulations for each solver
        for solver_type in solvers
            println("\nRunning simulations with $solver_type solver...")
            
            # Use appropriate POMDP type
            problem = uppercase(solver_type) == "MOMDP_SARSOP" ? momdp : pomdp
            
            # Run all simulations
            sim_results = simulate_many(
                problem,
                policies[solver_type],
                num_simulations,
                collect_beliefs=true,  # Collect beliefs for detailed plots
                seed=seed,
                verbose=verbose,
                replay_data=replay_data[1:num_simulations]
            )
            
            # Add policy generation time
            sim_results["policy_solve_time"] = policy_times[solver_type]
            
            # Store results
            size_results[solver_type] = sim_results
        end
        
        # Save size results
        size_results_path = joinpath(experiment_dir, "results_$(size_name).json")
        save_results_to_json(size_results, size_results_path)
        
        all_results[size_name] = size_results
        
        # Generate detailed belief evolution plots for a subset of runs
        if num_detailed_plots > 0
            generate_detailed_belief_plots(
                size_results,
                pomdp,
                momdp,
                experiment_dir,
                size_name,
                num_detailed_plots
            )
        end
    end
    
    # Save combined results
    combined_results_path = joinpath(experiment_dir, "all_results.json")
    save_results_to_json(all_results, combined_results_path)
    
    println("\n" * "="^60)
    println("Experiment complete!")
    println("Results saved to: $experiment_dir")
    println("="^60)
    
    return experiment_dir, all_results
end

"""
Save results to JSON, excluding belief histories to keep file size manageable.
Belief histories are saved separately if needed.
"""
function save_results_to_json(results::Dict, filepath::String)
    # Deep copy to avoid modifying original
    save_data = deepcopy(results)
    
    # Remove belief histories from each simulation
    for (solver, solver_results) in save_data
        if haskey(solver_results, "simulation_metrics")
            for metrics in solver_results["simulation_metrics"]
                delete!(metrics, "belief_history")
            end
        end
    end
    
    open(filepath, "w") do f
        JSON.print(f, save_data, 4)
    end
    
    println("Results saved to: $filepath")
end

"""
Generate detailed belief evolution plots for a subset of runs.
"""
function generate_detailed_belief_plots(
    results::Dict,
    pomdp,
    momdp,
    output_dir::String,
    problem_size::String,
    num_plots::Int
)
    plots_dir = joinpath(output_dir, "belief_evolution_plots", problem_size)
    mkpath(plots_dir)
    
    # For each solver
    for (solver, solver_results) in results
        solver_dir = joinpath(plots_dir, solver)
        mkpath(solver_dir)
        
        # Use appropriate problem formulation
        problem = uppercase(solver) == "MOMDP_SARSOP" ? momdp : pomdp
        is_momdp = isa(problem, PlanningProblem)
        
        # Plot for first num_plots simulations
        for i in 1:min(num_plots, length(solver_results["simulation_metrics"]))
            sim_metrics = solver_results["simulation_metrics"][i]
            
            # Extract belief history if available
            belief_history = get(sim_metrics, "belief_history", nothing)
            if belief_history === nothing
                continue
            end
            
            # Get run details and true end time
            run_details = sim_metrics["iterations"]
            true_end_time = run_details[1]["Tt"]
            
            # Create 2D belief evolution plot with actions
            p = plot_2d_belief_evolution_with_actions(
                belief_history,
                run_details,
                true_end_time,
                sim_metrics["min_end_time"],
                sim_metrics["max_end_time"],
                title_prefix="$solver - Run $i - ",
                is_momdp=is_momdp
            )
            
            if p !== nothing
                savefig(p, joinpath(solver_dir, "belief_evolution_run_$(lpad(i, 3, '0')).png"))
            end
        end
    end
end

"""
Load problem size configurations from reference problems.
"""
function load_problem_configs(reference_dir::String="reference_problems")
    configs = Dict{String, Dict{String, Any}}()
    
    # Define problem sizes
    problem_definitions = Dict(
        "small" => Dict(
            "filename" => "std_div_3/problems_l_1_u_12_n_1000_s_42.json",
            "min_end_time" => 1,
            "max_end_time" => 12
        ),
        "medium" => Dict(
            "filename" => "std_div_3/problems_l_1_u_26_n_1000_s_42.json", 
            "min_end_time" => 1,
            "max_end_time" => 26
        ),
        "large" => Dict(
            "filename" => "std_div_3/problems_l_1_u_52_n_1000.json",
            "min_end_time" => 1,
            "max_end_time" => 52
        )
    )
    
    for (size_name, size_def) in problem_definitions
        filepath = joinpath(reference_dir, size_def["filename"])
        if !isfile(filepath)
            error("Reference file not found: $filepath")
        end
        
        configs[size_name] = Dict(
            "replay_data_path" => filepath,
            "min_end_time" => size_def["min_end_time"],
            "max_end_time" => size_def["max_end_time"]
        )
    end
    
    return configs
end