
# ============================================================================
# Common functions shared between POMDP and MOMDP formulations  
# ============================================================================

function calculate_reward(t::Int, Ta::Int, Tt::Int, action_time::Int, min_end_time::Int, max_end_time::Int; lambda_c::Real = 3.0, lambda_e::Real = 2.0, lambda_f::Real = 1000.0)
    
    # If the project is done, return neutral reward
    if t >= Tt || t + 1 == max_end_time
        return 0.0
    end

    # Simple reward - penalize for the difference between announced and true end time
    r = -lambda_c * abs(action_time - Tt)

    # Add penalty if action changes from previous announced time
    if t > 0 && Ta != action_time
        if (action_time != Tt)
            r -= lambda_e  # Penalty for changing the announced time
        end
    end

    # Heavy penalty for wrong announcement when project completes
    if Tt == t
        if action_time != Tt
            r -= lambda_f
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

    # Poisson observation: rate = remaining time, so observations narrow as project progresses
    λ = max(1, Tt - t)
    poisson_dist = Poisson(λ)

    # Compute probabilities: To = t + X where X ~ Poisson(λ), so P(To) = P(X = To - t)
    probs = Float64[]
    for To_val in possible_Tos
        x = To_val - t
        p = x >= 0 ? pdf(poisson_dist, x) : 0.0
        push!(probs, p)
    end

    # Normalize over the truncated range
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

# ============================================================================
# Stochastic Tt Transition
# ============================================================================

function compute_tt_transition_distribution(Ta::Int, action::Int, Tt::Int, max_end_time::Int;
        p_no_effect::Float64 = 0.4, p_small::Float64 = 0.5, delta_small::Int = 1,
        p_large::Float64 = 0.1, delta_large::Int = 3)
    # If the agent keeps the same announcement, Tt is unchanged
    if action == Ta
        return Deterministic(Tt)
    end

    # Agent changed its announcement (replanned) — Tt may increase
    candidates = [Tt, min(Tt + delta_small, max_end_time), min(Tt + delta_large, max_end_time)]
    raw_probs  = [p_no_effect, p_small, p_large]

    # Merge duplicates caused by clamping to max_end_time
    merged = Dict{Int, Float64}()
    for (v, p) in zip(candidates, raw_probs)
        merged[v] = get(merged, v, 0.0) + p
    end

    vals  = collect(keys(merged))
    probs = collect(values(merged))

    return SparseCat(vals, probs)
end