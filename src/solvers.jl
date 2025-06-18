# Enumerated type for selecting solver
@enum SolverType begin
    FIB
    PBVI
    POMCPOW
    QMDP
    NATIVE_SARSOP
    CXX_SARSOP
    POMCP
    MOSTLIKELY
    OBSERVEDTIME
end

struct MostLikelyPolicy <: Policy end

function POMDPs.action(policy::MostLikelyPolicy, belief, state=nothing)
    # Extract the most likely state from the belief
    states = belief.state_list
    probs = belief.b
    max_prob_index = argmax(probs)
    # s is the most likely state
    s = states[max_prob_index]

    # Unpack state variables
    t, Ta, To = s

    # To is the most likely true end time
    observed_time = To

    # Determine the action
    if observed_time != Ta
        return AnnounceAction(observed_time)
    else
        return AnnounceAction(Ta)
    end
end

struct ObservedTimePolicy <: Policy end

function POMDPs.action(policy::ObservedTimePolicy, belief, state=nothing)

    # If first stand we have a uniform belief over all possible end times - Pick the middle
    if isa(belief, POMDPTools.POMDPDistributions.SparseCat)
        end_times = [b[3] for b in belief.vals]  # Assuming the third column is the end time
        # Extract min and max end times
        min_end_time = minimum(end_times)  # Assuming the third column is the end time
        max_end_time = maximum(end_times)  # Assuming the third column is the end time

        # Calculate the middle end time
        middle_end_time = Int(round((min_end_time + max_end_time) / 2))
        
        Ta = middle_end_time
    else
        # If only one state, should be a tuple and the "observation" is the end time
        Ta = belief[3]
    end
    
    return AnnounceAction(Ta)
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
    elseif uppercase(solver_type) == "NATIVE_SARSOP"
        println("Computing policy using NativeSARSOP solver")
        elapsed_time = @elapsed policy = solve(NativeSARSOP.SARSOPSolver(), pomdp)
    elseif uppercase(solver_type) == "CXX_SARSOP"
        println("Computing policy using SARSOP solver")
        elapsed_time = @elapsed policy = solve(SARSOP.SARSOPSolver(timeout=600), pomdp) # use precision=, timeout= to change exit criterion, policy_interval= (seconds)
    elseif uppercase(solver_type) == "POMCP"
        println("Computing policy using POMCP solver")
        elapsed_time = @elapsed policy = solve(POMCPSolver(), pomdp)
    elseif uppercase(solver_type) == "MOSTLIKELY"
        elapsed_time = @elapsed policy = MostLikelyPolicy()
    elseif uppercase(solver_type) == "OBSERVEDTIME"
        elapsed_time = @elapsed policy = ObservedTimePolicy()
    else
        println("Error: Invalid solver type: $solver_type.")
        exit(1)
        # println("Invalid solver type: $solver_type. Using random policy by default.")
        # elapsed_time = @elapsed policy = RandomPolicy(pomdp)
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