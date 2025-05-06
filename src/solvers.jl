# Enumerated type for solver types


function get_policy(pomdp, solver_type, output_dir; verbose::Bool=false)
    if solver_type == "fib"
        println("Computing policy using FIB solver")
        elapsed_time = @elapsed policy = solve(FIBSolver(), pomdp)
    elseif solver_type == "pbvi"
        println("Computing policy using PBVI solver")
        elapsed_time = @elapsed policy = solve(PBVISolver(), pomdp)
    elseif solver_type == "pomcpow"
        println("Computing policy using POMCPOW solver")
        elapsed_time = @elapsed policy = solve(POMCPOWSolver(), pomdp) # How should I adjust criterion
    elseif solver_type == "qmdp"
        println("Computing policy using QMDP solver")
        elapsed_time = @elapsed policy = solve(QMDPSolver(), pomdp)
    elseif solver_type == "sarsop"
        println("Computing policy using SARSOP solver")
        elapsed_time = @elapsed policy = solve(SARSOPSolver(), pomdp)
    # elseif solver_type == "mostlikely"
    #     elapsed_time = @elapsed policy = MostLikelyPolicy()
    else
        println("Invalid solver type: $solver_type. Using random policy by default.")
        elapsed_time = @elapsed policy = RandomPolicy(pomdp)
    end

    output_filepath = joinpath(output_dir, "policy.jld2")
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    if isfile(output_filepath)
        if verbose
            println("Removing existing policy file: $output_filepath")
        end
        rm(output_filepath)
    end

    save(output_filepath, "policy", policy)

    if verbose
        println("Policy saved to: $output_filepath")
    end

    println("Time to compute policy: ", elapsed_time, " seconds")
    output = Dict(
        "policy" => policy,
        "comp_policy_time" => elapsed_time
    )
    return output
end