module POMDPPlanning

export main

using ArgParse
using PkgVersion

# POMDP Modelings
using POMDPs
using QuickPOMDPs
using POMDPTools
using JLD2

# Solvers
using POMCPOW
using QMDP
using FIB
using PointBasedValueIteration
using NativeSARSOP

# Utilities
using Distributions
using Random
using LinearAlgebra
using Plots
using JSON
using Statistics
using ProgressMeter
using Dates

const VERSION = string(PkgVersion.@Version)

# Include subfiles The import order matters for compilation
include("utils.jl")
include("problem.jl")
include("solvers.jl")
include("simulation.jl")
include("analysis.jl")
include("experiments.jl")


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
        "--solver", "-s"
            help = "Solver type to use (e.g., random, fib, pbvi, pomcpow, qmdp, sarsop)"
            arg_type = String
            default = "default_value"
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
            default = 0.99
        "--verbose", "-v"
            help = "Enable verbose output"
            nargs = 0
        "--debug", "-D"
            help = "Enable debug output"
            nargs = 0
        "--output-dir", "-o"
            help = "Directory to save output files"
            arg_type = String
            default = "output"
        "--policy-file", "-p"
            help = "Path to policy file (.jld2) for evaluation"
            arg_type = String
        "--true-end-time", "-t"
            help = "Fixed true end time for evaluation (if not provided, random values will be used)"
            arg_type = Int
        "--initial-announce", "-a"
            help = "Initial announced time (default is min_end_time from policy metadata)"
            arg_type = Int
        "command"
            help = "Command to execute (solve or evaluate)"
            required = true
    end
    
    return parse_args(s)
end

# Main function to run the CLI
function main()
    args = parse_commandline()
    
    # Add your CLI logic here
    if args["command"] == "solve"
        if args["debug"]
            println("Running with: $(args)♮")
        end

        # Check if the min-end-time is less than the max-end-time
        if args["min-end-time"] > args["max-end-time"]
            println("Error: min-end-time must be less than or equal to max-end-time")
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

        # Check that the solver type is one of the valid options (defined by the enum SolverType in solvers.jl)
        if !(uppercase(args["solver"]) in collect(string.(instances(SolverType))))
            println("Error: invalid solver type. Valid options are: $(join(string.(instances(MyEnum)), ", "))")
            return 1
        end

        

        # If a seed is provided, set the random seed
        if args["seed"] != nothing
            println("Setting random seed to $(args["seed"])")
            Random.seed!(args["seed"])
        end

        # Create planning POMDP
        pomdp = define_pomdp(
            args["min-end-time"],
            args["max-end-time"],
            args["discount"],
            verbose=args["verbose"]
        )

        policy = get_policy(
            pomdp,
            args["solver"],
            args["output-dir"],
            verbose=args["verbose"]
        )

    elseif args["command"] == "evaluate"
        if args["policy-file"] === nothing
            println("Error: policy file is required for evaluation")
            return 1
        end
        
        # Load policy and metadata
        if !isfile(args["policy-file"])
            println("Error: policy file not found: $(args["policy-file"])")
            return 1
        end
        
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
        
        # Create POMDP for evaluation
        pomdp = define_pomdp(
            min_end_time,
            max_end_time,
            discount_factor,
            verbose=args["verbose"]
        )
        
        # Run evaluation
        evaluate_policy(
            pomdp, 
            policy, 
            args["num_simulations"], 
            args["output-dir"],
            fixed_true_end_time=args["true-end-time"],
            initial_announce=args["initial-announce"],
            verbose=args["verbose"]
        )
        
    else
        # Handle unknown command
        println("Unknown command: $(args["command"])")
    end
    
    return 0
end

end # module