# Enumerated type for selecting solver
@enum SolverType begin
    FIB
    PBVI
    POMCPOW
    QMDP
    SARSOP
end


function get_policy(pomdp, solver_type, output_dir;
                    min_end_time::Int=10, 
                    max_end_time::Int=20, 
                    discount_factor::Float64=0.99,
                    verbose::Bool=false)

    if uppercase(solver_type) == "FIB"
        println("Computing policy using FIB solver")
        elapsed_time = @elapsed policy = solve(FIBSolver(), pomdp)
    elseif uppercase(solver_type) == "PBVI"
        println("Computing policy using PBVI solver")
        elapsed_time = @elapsed policy = solve(PBVISolver(), pomdp)
    elseif uppercase(solver_type) == "POMCPOW"
        println("Computing policy using POMCPOW solver")
        elapsed_time = @elapsed policy = solve(POMCPOWSolver(), pomdp) # How should I adjust criterion
    elseif uppercase(solver_type) == "QMDP"
        println("Computing policy using QMDP solver")
        elapsed_time = @elapsed policy = solve(QMDPSolver(), pomdp)
    elseif uppercase(solver_type) == "SARSOP"
        println("Computing policy using SARSOP solver")
        elapsed_time = @elapsed policy = solve(SARSOPSolver(), pomdp)
    # elseif solver_type == "mostlikely"
    #     elapsed_time = @elapsed policy = MostLikelyPolicy()
    else
        println("Invalid solver type: $solver_type. Using random policy by default.")
        elapsed_time = @elapsed policy = RandomPolicy(pomdp)
    end
    
    # Create directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Create metadata
    metadata = Dict(
        "min_end_time" => min_end_time,
        "max_end_time" => max_end_time,
        "solver_type" => solver_type,
        "discount_factor" => discount_factor,
        "generation_date" => string(Dates.now()),
        "computation_time" => elapsed_time
    )
    
    # Save policy
    policy_filepath = joinpath(output_dir, "policy_$(lowercase(solver_type)).jld2")
    if isfile(policy_filepath)
        if verbose
            println("Removing existing policy file: $policy_filepath")
        end
        rm(policy_filepath)
    end
    save(policy_filepath, "policy", policy, "metadata", metadata)
    
    # Save metadata separately as JSON for easier inspection
    metadata_filepath = joinpath(output_dir, "policy_$(lowercase(solver_type)).json")
    open(metadata_filepath, "w") do f
        JSON.print(f, metadata, 4)  # 4 spaces for indentation
    end
    
    if verbose
        println("Policy saved to: $policy_filepath")
        println("Metadata saved to: $metadata_filepath")
    end

    println("Time to compute policy: ", elapsed_time, " seconds")
    
    output = Dict(
        "policy" => policy,
        "metadata" => metadata,
        "comp_policy_time" => elapsed_time
    )
    
    return output
end