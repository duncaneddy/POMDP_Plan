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
            arg_type = Bool
            default = false
            nargs = 0
        "--output-dir", "-o"
            help = "Directory to save output files"
            arg_type = String
            default = "output"
        "command"
            help = "Command to execute"
            required = true
    end
    
    return parse_args(s)
end

# Main function to run the CLI
function main()
    args = parse_commandline()
    
    # Add your CLI logic here
    if args["command"] == "solve"
        if args["verbose"]
            println("Running with: $(args)♮")
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

    else
        println("Unknown command: $(args["command"])")
    end
    
    return 0
end

end # module