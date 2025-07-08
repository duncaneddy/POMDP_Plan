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