
# ============================================================================
# Common functions shared between POMDP and MOMDP formulations  
# ============================================================================

function calculate_reward(t::Int, Ta::Int, Tt::Int, action_time::Int, min_end_time::Int, max_end_time::Int)
    # If the project is done, return neutral reward
    if t >= Tt || t + 1 == max_end_time
        return 0.0
    end

    # Simple reward - penalize for the difference between announced and true end time
    r = -2 * abs(action_time - Tt)

    # Add penalty if action changes from previous announced time
    if t > 0 && Ta != action_time
        if (action_time != Tt)
            r -= 3.0  # Penalty for changing the announced time
        end
    end

    # Heavy penalty for wrong announcement when project completes
    if Tt == t
        if action_time != Tt
            r -= 1000
        end
    end

    return r
end

# ============================================================================
# Common Observation Functions
# ============================================================================

function compute_observation_distribution(t::Int, Tt::Int, min_end_time::Int, max_end_time::Int; std_divisor::Real = 3)
    # Check for deterministic cases
    if t >= Tt || t + 1 == max_end_time || Tt - t <= 0
        return nothing  # Caller should handle deterministic case
    end

    # Compute observation range
    min_obs_time = max(t + 1, min_end_time)
    possible_Tos = collect(min_obs_time:max_end_time)
    
    # Calculate truncated normal parameters
    μ = Tt
    σ = (Tt - t) / std_divisor
    
    base_dist = Normal(μ, σ)
    
    # Compute truncation bounds
    lower = min_obs_time
    upper = max_end_time
    cdf_lower = cdf(base_dist, lower)
    cdf_upper = cdf(base_dist, upper)
    denom = cdf_upper - cdf_lower
    
    # Handle edge case where denominator is zero
    if denom ≈ 0.0
        return nothing  # Fallback to deterministic
    end
    
    # Compute probabilities for each possible observation
    probs = Float64[]
    for To_val in possible_Tos
        p = pdf(base_dist, To_val) / denom
        push!(probs, p)
    end
    
    # Normalize probabilities (safety check)
    total_p = sum(probs)
    if total_p ≈ 0.0
        return nothing  # Fallback to deterministic
    end
    probs ./= total_p
    
    return (possible_Tos, probs)
end

function create_momdp_observation(t::Int, Tt::Int, min_end_time::Int, max_end_time::Int; std_divisor::Real = 3)
    # Try to compute stochastic distribution
    result = compute_observation_distribution(t, Tt, min_end_time, max_end_time; std_divisor=std_divisor)
    
    if result === nothing
        # Deterministic case: return true end time
        return Deterministic(Tt)
    else
        possible_Tos, probs = result
        return SparseCat(possible_Tos, probs)
    end
end

function create_pomdp_observation(t::Int, Ta::Int, Tt::Int, min_end_time::Int, max_end_time::Int; std_divisor::Real = 3)
    # Try to compute stochastic distribution
    result = compute_observation_distribution(t, Tt, min_end_time, max_end_time; std_divisor=std_divisor)
    
    if result === nothing
        # Deterministic case: return full observation tuple with true end time
        return Deterministic((t, Ta, Tt))
    else
        possible_Tos, probs = result
        # Create observation tuples for each possible observed time
        obs_list = [(t, Ta, To_val) for To_val in possible_Tos]
        return SparseCat(obs_list, probs)
    end
end