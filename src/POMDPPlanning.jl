module POMDPPlanning

export main

using ArgParse
using PkgVersion

# POMDP Modelings
using POMDPs
using QuickPOMDPs
using POMDPTools
using MOMDPs
using JLD2

# Solvers
using POMCPOW
using QMDP
using FIB
using PointBasedValueIteration
import NativeSARSOP
import SARSOP
using BasicPOMCP

# Utilities
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using JSON
using Statistics
using ProgressMeter
using Dates

const VERSION = string(PkgVersion.@Version)

# Include subfiles The import order matters for compilation
include("utils.jl")
include("problem_common.jl")
include("problem.jl")
include("momdp.jl")
include("solvers.jl")
include("simulation.jl")
include("analysis.jl")
include("experiments.jl")
include("experiments_paper.jl")


function parse_commandline()
    s = ArgParseSettings(
        description = "POMDPPlanning CLI",
        add_version = true,
        version = VERSION,
        add_help = true,
    )
    
    @add_arg_table! s begin
        "--seed", "-r"
            help = "Random seed for reproducibility"
            arg_type = Int
        "--solvers", "-s"
            help = "Comma-separated list of solvers (e.g FIG, PVBI, POMCPOW, QMDP, SARSOP) to compare in experiments, or 'all' for all solvers"
            arg_type = String
            default = "all"
        "--num_simulations", "-n"
            help = "Number of simulations to run"
            arg_type = Int
            default = 100
        "--min-end-time", "-l"
            help = "Minimum end time for the project"
            arg_type = Int
            default = 10
        "--max-end-time", "-u"
            help = "Maximum end time for the project"
            arg_type = Int
            default = 20
        "--discount", "-d"
            help = "Discount factor for the POMDP"
            arg_type = Float64
            default = 0.98 # Keep fairly high since this is actually a finite horizon problem
        "--std-divisor", "-i"
            help = "Standard deviation divisor for the noise in the true end time observation (default is 3.0)"
            arg_type = Float64
            default = 3.0
        "--verbose", "-v"
            help = "Enable verbose output"
            nargs = 0
        "--no-plot"
            help = "Disable plotting of results"
            nargs = 0
        "--debug", "-D"
            help = "Enable debug output"
            nargs = 0
        "--output-dir", "-o"
            help = "Directory to save output files"
            arg_type = String
            default = "results"
        "--policy-file", "-p"
            help = "Path to policy file (.jld2) for evaluation"
            arg_type = String
        "--true-end-time", "-t"
            help = "Fixed true end time for evaluation (if not provided, random values will be used)"
            arg_type = Int
            default = nothing
        "--initial-announce", "-a"
            help = "Initial announced time (default is min_end_time from policy metadata)"
            arg_type = Int
            default = nothing
        "--replay-data", "-z"
            help = "Path to an evaluation_results.json file to replay simulation data for reproduction"
            arg_type = String
        "--policy-timeout"
            help = "Timeout for policy generation in seconds (default is 300 seconds)"
            arg_type = Int
            default = 300
        "command"
            help = "Command to execute (solve or evaluate)"
            required = true
    end
    
    return parse_args(s)
end

# Main function to run the CLI
function main()
    args = parse_commandline()

    if args["debug"]
        println("Running with: $(args)♮")
    end

    # Check if the min-end-time is less than the max-end-time
    if args["min-end-time"] > args["max-end-time"]
        println("Error: min-end-time must be less than or equal to max-end-time")
        return 1
    end

    if args["initial-announce"] !== nothing && (args["initial-announce"] < args["min-end-time"] || args["initial-announce"] > args["max-end-time"])
        println("Error: initial-announce must be between min-end-time and max-end-time")
        return 1
    end

    # Check if the discount factor is between 0 and 1
    if args["discount"] <= 0 || args["discount"] >= 1
        println("Error: discount factor must be between 0 and 1")
        return 1
    end

    # Check if the number of simulations is positive
    if args["num_simulations"] <= 0
        println("Error: number of simulations must be positive")
        return 1
    end

    # Parse solvers list
    solvers_str = args["solvers"]
    if solvers_str == "all"
        solvers = string.(collect(instances(SolverType)))
    else
        solvers = string.(split(solvers_str, ","))
    end

    # Validate each solver
    for solver in solvers
        if !(uppercase(solver) in collect(string.(instances(SolverType))))
            println("Error: invalid solver type: $solver. Valid options are: $(join(string.(instances(SolverType)), ", ")) or 'all'")
            return 1
        end
    end
    
    # If a seed is provided, set the random seed
    if args["seed"] != nothing
        println("Setting random seed to $(args["seed"])")
        Random.seed!(args["seed"])
    end
    
    # Add your CLI logic here
    if args["command"] == "solve"

        # Create planning problem
        pomdp = nothing

        solvers = split(solvers_str, ",")

        for solver in solvers
            
            if args["verbose"]
                println("Running solver: $solver")
            end

            if uppercase(solver) == "MOMDP_SARSOP"
                # Create MOMDP POMDP
                pomdp = define_momdp(
                    args["min-end-time"],
                    args["max-end-time"],
                    args["discount"],
                    initial_announce=args["initial-announce"],
                    std_divisor=args["std-divisor"]
                )
            elseif pomdp === nothing
                # Create planning POMDP
                pomdp = define_pomdp(
                    args["min-end-time"],
                    args["max-end-time"],
                    args["discount"],
                    verbose=args["verbose"],
                    initial_announce=args["initial-announce"],
                    fixed_true_end_time=args["true-end-time"],
                    std_divisor=args["std-divisor"]
                )
            end
            
            # Solve the POMDP using the specified solver
            get_policy(
                pomdp,
                solver,
                args["output-dir"],
                verbose=args["verbose"],
                policy_timeout=args["policy-timeout"]
            )
        end


    elseif args["command"] == "evaluate"

        # Create planning POMDP
        if uppercase(solvers[1]) == "MOMDP_SARSOP"
            # Create MOMDP POMDP
            pomdp = define_momdp(
                args["min-end-time"],
                args["max-end-time"],
                args["discount"],
                initial_announce=args["initial-announce"],
                std_divisor=args["std-divisor"]
            )
        else
            # Create planning POMDP
            pomdp = define_pomdp(
                args["min-end-time"],
                args["max-end-time"],
                args["discount"],
                verbose=args["verbose"],
                initial_announce=args["initial-announce"],
                fixed_true_end_time=args["true-end-time"],
                std_divisor=args["std-divisor"]
            )
        end
        
        # Load policy file or create one
        if args["policy-file"] == nothing

            if args["solvers"] == nothing
                println("Error: no solvers specified for policy generation")
                return 1
            end

            solvers = split(solvers_str, ",")

            if length(solvers) != 1
                println("Error: only one solver can be specified for policy generation")
                return 1
            end
            solver = solvers[1]

            if solver == "all"
                println("Error: 'all' is not a valid solver for policy generation")
                return 1
            end


            if args["verbose"]
                println("Running solver: $solver")
            end
            
            # Solve the POMDP using the specified solver
            policy_output = get_policy(
                pomdp,
                solver,
                args["output-dir"],
                verbose=args["verbose"],
                policy_timeout=args["policy-timeout"]
            )
            
            policy = policy_output["policy"]
        else
        
            loaded_data = load(args["policy-file"])
            policy = loaded_data["policy"]
            metadata = loaded_data["metadata"]
            
            if args["debug"]
                println("Loaded policy from $(args["policy-file"])")
                println("Metadata: $metadata")
            end
            
            # Create POMDP with parameters from metadata
            min_end_time = metadata["min_end_time"]
            max_end_time = metadata["max_end_time"]
            discount_factor = metadata["discount_factor"]
            
            # Override with command line arguments if provided
            if args["min-end-time"] !== nothing
                min_end_time = args["min-end-time"]
            end
            if args["max-end-time"] !== nothing
                max_end_time = args["max-end-time"]
            end
            if args["discount"] !== nothing
                discount_factor = args["discount"]
            end
            
            # Validate parameter compatibility
            if min_end_time != metadata["min_end_time"] || max_end_time != metadata["max_end_time"]
                println("Warning: Evaluation parameters don't match policy metadata")
                println("  Policy: min_end_time=$(metadata["min_end_time"]), max_end_time=$(metadata["max_end_time"])")
                println("  Evaluation: min_end_time=$min_end_time, max_end_time=$max_end_time")
                
                confirm_continue = false
                while !confirm_continue
                    print("Continue anyway? (y/n): ")
                    response = lowercase(readline())
                    if response in ["y", "yes"]
                        confirm_continue = true
                    elseif response in ["n", "no"]
                        println("Evaluation cancelled")
                        return 0
                    end
                end
            end
        end

        # Load simulation data if provided
        replay_data = nothing
        if args["replay-data"] !== nothing
            println("Loading replay data from $(args["replay-data"])")
            try
                replay_data = JSON.parsefile(args["replay-data"])

                # Check if "simulation_data" key exists
                if haskey(replay_data, "simulation_data")
                    # Convert to array of dictionaries for easier access
                    replay_data = replay_data["simulation_data"]
                else
                    println("Warning: 'simulation_data' key not found when replay data was provided. This means that the replay data is not in the expected format. Please ensure that the evaluation_results.json file contains a 'simulation_data' key with an array of simulation results.")
                    return 1
                end
            catch e
                println("Error loading replay data: $(e)")
                return 1
            end
        end

        # Print type of policy
        isa(policy, ObservedTimePolicy) && println("This is an ObservedTimePolicy")
        
        # Run evaluation
        evaluate_policy(
            pomdp, 
            policy, 
            args["num_simulations"], 
            args["output-dir"],
            fixed_true_end_time=args["true-end-time"],
            initial_announce=args["initial-announce"],
            seed=args["seed"],  # Add this line
            verbose=args["verbose"],
            debug=args["debug"],
            solver=args["solvers"],  # Pass the solver type for metadata
            replay_data=replay_data  # Pass the replay data if provided
        )

    elseif args["command"] == "experiments"
        if args["debug"]
            println("Running experiment with: $(args)")
        end

        if args["num_simulations"] <= 0
            println("Error: number of simulations must be positive")
            return 1
        end

        # Run the experiment
        run_experiment(
            args["min-end-time"],
            args["max-end-time"],
            solvers,
            args["num_simulations"],
            args["output-dir"],
            fixed_true_end_time=args["true-end-time"],
            initial_announce=args["initial-announce"],
            discount_factor=args["discount"],
            seed=args["seed"],  # Add this line
            verbose=args["verbose"],
            std_divisor=args["std-divisor"]
        )
        
    else
        # Handle unknown command
        println("Unknown command: $(args["command"])")
    end
    
    return 0
end

end # module