# Define a structured action type for announcing a specific time
struct AnnounceAction
    announced_time::Int
end

function define_pomdp(min_end_time::Int, max_end_time::Int, discount_factor::Float64; initial_announce::Union{Int, Nothing}=nothing, fixed_true_end_time::Union{Int, Nothing}=nothing, verbose::Bool = false)
    # Constants for rewards
    IMPOSSIBLE_TIME_REWARD = -1000
    WRONG_END_TIME_REWARD = -1000

    # Define the range of possible end times
    if verbose
        println("Defining POMDP with:\n- Min End Time: $min_end_time\n- Max End Time: $max_end_time\n- Discount Factor: $discount_factor")
    end
    

    # Define actions: an AnnounceAction for each possible observed time
    actions = [AnnounceAction(To_val) for To_val in min_end_time:max_end_time]

    pomdp = QuickPOMDP(
        states = [(t, Ta, Tt) for t in 0:max_end_time,
                                 Ta in min_end_time:max_end_time,
                                 Tt in min_end_time:max_end_time],
        actions = actions,
        actiontype = AnnounceAction,
        observations = [(t, Ta, To) for t in 0:max_end_time,
                                     Ta in min_end_time:max_end_time,
                                     To in min_end_time:max_end_time],

        discount = discount_factor,

        transition = function(s, a)
            t, Ta, Tt = s
            
            # Move time forward, but not beyond the true end time
            t = min(t + 1, Tt)

            # Update Ta to the announced_time chosen by the action
            # Note that the action can be any number in min_end_time:max_end_time 
            # The paper restricts this to only the previous observed time
            new_Ta = a.announced_time

            sp = (t, new_Ta, Tt)

            return Deterministic(sp)
        end,

        observation = function(a, sp)
            # We have just transitioned from (t - 1, Ta_prev, Tt) to (t, Ta, Tt)
            t, Ta, Tt = sp

            # Debug Step 1: If we have deterministic observation that provide the true end time, does the action announce
            # the true end time.
            # Answer: Yes, immediately after the first observation the announce time will be the true end time and stay there
            # return Deterministic((t, Ta, Tt))

            # Debug Step 2: We try the same thing but with just a single SparseCat observation
            # This should be the same as the deterministic observation.
            # Answer: Yes, this works as expected and gives the same results as above
            # return SparseCat([(t, Ta, Tt)], [1.0])

            # Debug Step 3: Now we try with a simple, but heavily weighted observation function
            # This funciton allows for observation of the true end time and potentially other times 
            # one timestep before or after the true end time. The true end time is heavily weighted
            # as the most likely observation.
            # 
            # Result: 

            obs_list = [(t, Ta, Tt)]

            # Account for the fact that the true end time might be at the minimum or maximum end time
            # And so we don't add observations that are out of bounds
            if Tt > min_end_time
                obs_list = push!(obs_list, (t, Ta, Tt - 1))
            end

            if Tt < max_end_time
                obs_list = push!(obs_list, (t, Ta, Tt + 1))
            end

            # Define probabilities for the observations
            if length(obs_list) == 1
                probs = [1.0]
            elseif length(obs_list) == 2
                # If we have two observations, we can assign equal probabilities
                probs = [0.95, 0.05]
            else
                # If we have three observations, assign equal probabilities
                probs = [0.9, 0.05, 0.05]
            end

            return SparseCat(obs_list, probs)

        end,

        reward = function(s, a)
            # Reward for taking action a in state s
            t, Ta, Tt = s

            # Dead-simple reward - just penalize for the difference between announced and true end time
            r = -1 * abs(Ta - Tt)

            # This was the original reward logic of penalizing based on the action (announced time) vs the true end time
            # but it has some strange behavior
            # r = -1 * abs(a.announced_time - Tt)

            return r
        end,

        initialstate = function()
            if initial_announce == nothing
                initial_announce = min_end_time
            end

            if isnothing(fixed_true_end_time)
                # Randomly select a true end time
                possible_states = [(0, initial_announce, Tt) for Tt in min_end_time:max_end_time]
            else
                # Use the fixed true end time
                possible_states = [(0, initial_announce, fixed_true_end_time)]
            end
            
            num_states = length(possible_states)

            probabilities = fill(1.0 / num_states, num_states)

            return SparseCat(possible_states, probabilities)
        end,

        isterminal = function(s)
            t, Ta, Tt = s
            return t == Tt + 1
        end
    )
    return pomdp
end

# function define_pomdp(min_end_time::Int, max_end_time::Int, discount_factor::Float64; initial_announce::Union{Int, Nothing}=nothing, fixed_true_end_time::Union{Int, Nothing}=nothing, verbose::Bool = false)
#     # Constants for rewards
#     IMPOSSIBLE_TIME_REWARD = -1000
#     WRONG_END_TIME_REWARD = -1000

#     # Define the range of possible end times
#     if verbose
#         println("Defining POMDP with:\n- Min End Time: $min_end_time\n- Max End Time: $max_end_time\n- Discount Factor: $discount_factor")
#     end
    

#     # Define actions: an AnnounceAction for each possible observed time
#     actions = [AnnounceAction(To_val) for To_val in min_end_time:max_end_time]

#     pomdp = QuickPOMDP(
#         states = [(t, Ta, Tt) for t in 0:max_end_time,
#                                  Ta in min_end_time:max_end_time,
#                                  Tt in min_end_time:max_end_time],
#         actions = actions,
#         actiontype = AnnounceAction,
#         observations = [(t, Ta, To) for t in 0:max_end_time,
#                                      Ta in min_end_time:max_end_time,
#                                      To in min_end_time:max_end_time],

#         discount = discount_factor,

#         transition = function(s, a)
#             t, Ta, Tt = s
#             # Move time forward, but not beyond the true end time
#             t = min(t + 1, Tt)

#             # Update Ta to the announced_time chosen by the action
#             # Note that the action can be any number in min_end_time:max_end_time 
#             # The paper restricts this to only the previous observed time
#             new_Ta = a.announced_time
#             sp = (t, new_Ta, Tt)

#             return Deterministic(sp)
#         end,

#         observation = function(a, sp)
#             # We have just transitioned from (t - 1, Ta_prev, Tt) to (t, Ta, Tt)
#             t, Ta, Tt = sp

#             # If the project is done or we are at the timestep before the maximum project completion time
#             # just return Tt deterministically
#             if t >= Tt || t + 1 == max_end_time
#                 return Deterministic((t, Ta, Tt))
#             end

#             # Otherwise, we have the case where the project is not done yet
#             # The minimum completion time we can observe the maximium of 
#             # the current time plus 1 (i.e. we think that after the next transition we will be done)
#             # and the minimum observed completion time (min_end_time)
#             min_obs_time = max(t + 1, min_end_time)

#             possible_Tos = collect(min_obs_time:max_end_time)

#             # Calculate parameters of the truncated normal
#             mu = Tt
#             std = (Tt - t) / 1.5
            
#             if Tt - t <= 0 
#                 # The task is done, so we should observe the true end time
#                 return Deterministic((t, Ta, Tt))
#             end

#             base_dist = Normal(mu, std)
#             # the truncated normal distribution is defined from t+1 to max_end_time
#             lower = min_obs_time
#             upper = max_end_time
#             cdf_lower = cdf(base_dist, lower)
#             cdf_upper = cdf(base_dist, upper)
#             denom = cdf_upper - cdf_lower

#             probs = Float64[]
#             for To_val in possible_Tos
#                 p = (pdf(base_dist, To_val) / denom)
#                 push!(probs, p)
#             end

#             total_p = sum(probs)
#             if total_p == 0.0
#                 return Deterministic((t, Ta, Tt))
#             end
#             probs ./= total_p

#             obs_list = [(t, Ta, To_val) for To_val in possible_Tos]
            
#             return SparseCat(obs_list, probs)
#         end,

#         reward = function(s, a)
#             # Reward for taking action a in state s
#             t, Ta, Tt = s
#             # println("Ta", Ta)
#             # println("a.announced_time", a.announced_time)
#             earlier = -30
#             later = -45
#             incentive = 1
#             r = 0

#             # If announcing an impossible time
#             # Currently, time t + 1
#             # After time t - 1, we announced Ta
#             # Then, after time t, we announced a.announced_time
#             # Now we must announce a time >= t and, 
#             if (a.announced_time < t)
#                 return IMPOSSIBLE_TIME_REWARD
#             end

#             if Tt == t
#                 # if Tt = t, we must announce t = Tt
#                 if a.announced_time != Tt
#                     return WRONG_END_TIME_REWARD
#                 end
#                 return 0
#             end

#             if a.announced_time == Ta # not updating your announcement
#                 r = -1 * abs(Ta - Tt) # penalize for the difference between announced and true end time
#                 # Reward below was creating high rewards compared to announcing a new time, and the policy chooses to not change its announcement. I changed the reward to match the cases announcing a new time minus some constant
#                 # diff_announced = abs(a.announced_time - Tt) # difference between announced and true end time (probably near 1 or 2)
#                 # time_to_end = Tt - t # Will never be 0 because we check for Tt == t above
#                 # if a.announced_time < Tt
#                 #     r += earlier * (1 / time_to_end) * diff_announced 
#                 # elseif a.announced_time > Tt
#                 #     r += later * (1 / time_to_end) * diff_announced
#                 # end
#                 # r += incentive
#             end
            
#             if a.announced_time != Ta  # announcing a new time
#                 diff_announced = abs(Ta - a.announced_time) # difference between announced and true end time (probably near 1 or 2)                
#                 time_to_end = Tt - t # Will never be 0 because we check for Tt == t above
#                 # if Ta < Tt
#                 #     r += earlier * (1 / time_to_end) * diff_announced 
#                 # elseif Ta > Tt
#                 #     r += later * (1 / time_to_end) * diff_announced
#                 # end
#                 # Logic above was the definition before but Ta is the previously announced time, not the current action 
#                 if a.announced_time <= Tt
#                     r += earlier * (1 / time_to_end) * diff_announced 
#                 elseif a.announced_time > Tt
#                     r += later * (1 / time_to_end) * diff_announced
#                 end
#             end
#             return r
#         end,

#         initialstate = function()
#             if initial_announce == nothing
#                 initial_announce = min_end_time
#             end

#             if isnothing(fixed_true_end_time)
#                 # Randomly select a true end time
#                 possible_states = [(0, initial_announce, Tt) for Tt in min_end_time:max_end_time]
#             else
#                 # Use the fixed true end time
#                 possible_states = [(0, initial_announce, fixed_true_end_time)]
#             end
            
#             num_states = length(possible_states)
#             probabilities = fill(1.0 / num_states, num_states)
#             return SparseCat(possible_states, probabilities)
#         end,
#         isterminal = function(s)
#             t, Ta, Tt = s
#             return t == Tt + 1
#         end
#     )
#     return pomdp
# end