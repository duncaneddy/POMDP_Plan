# Define a structured action type for announcing a specific time
struct AnnounceAction
    announced_time::Int
end

# function define_pomdp(min_end_time::Int, max_end_time::Int, discount_factor::Float64; 
#                               initial_announce::Union{Int, Nothing}=nothing, 
#                               fixed_true_end_time::Union{Int, Nothing}=nothing, 
#                               verbose::Bool = false,
#                               penalty_config::Dict = Dict())

#     # Merge default penalties with user-provided config
#     default_penalties = Dict(
#         # Announcing time in the past
#         "impossible_time_reward" => -1000,
#         # Wrong announcement when project completes
#         "wrong_end_time_reward" => -1000,
#         # Reached announced deadline but project not done
#         "missed_deadline_penalty" => -1,
#         # Per time step of announcing later than completion
#         "over_commitment_penalty" => -15,
#         # Per step penalty for error in current estimate
#         "accuracy_penalty_rate" => -0.5,
#         # Flat cost for any change (always applied)
#         "base_change_penalty" => 10.0,
#         # Per step of change magnitude
#         "magnitude_penalty_rate" => 1.0,
#         # High penalty for step-before changes
#         "step_before_penalty" => 75.0,
#         # Base urgency penalty coefficient
#         "timing_penalty_rate" => 20.0,
#         # How much magnitude affects timing penalty
#         "magnitude_scaling" => 0.5,
#         # Additional penalty for announcing too early
#         "early_bias_penalty" => 0.0, # 12
#         # Additional penalty for announcing too late
#         "late_bias_penalty" => 0.0 # 8
#     )
    
#     penalties = merge(default_penalties, penalty_config)
    
#     if verbose
#         println("Defining Enhanced POMDP with:")
#         println("- Min End Time: $min_end_time")
#         println("- Max End Time: $max_end_time") 
#         println("- Discount Factor: $discount_factor")
#         println("- Penalty Configuration: $penalties")
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
#             t, Ta, Tt = s
#             new_Ta = a.announced_time
            
#             # === BASIC BEHAVIOR ===
#             if new_Ta < t
#                 return penalties["impossible_time_reward"]
#             end
            
#             reward = 0.0
            
#             # === IMMEDIATE CONSEQUENCES ===
            
#             # 1. Missed deadline: reached announced time but project not done
#             if t == Ta && t < Tt
#                 reward += penalties["missed_deadline_penalty"]
#             end
            
#             # # 2. Over-commitment: project finished but announced later
#             # if t == Tt && Ta > Tt
#             #     over_commit_days = Ta - Tt
#             #     reward += penalties["over_commitment_penalty"] * over_commit_days
#             # end
            
#             # === TERMINAL CONDITIONS ===
            
#             if t == Tt  # Project actually completes
#                 if new_Ta != Tt
#                     return penalties["wrong_end_time_reward"]
#                 else
#                     return reward  # Return any immediate consequences
#                 end
#             end
            
#             # === ONGOING PENALTIES ===
            
#             # 3. Accuracy penalty
#             accuracy_error = abs(new_Ta - Tt)
#             accuracy_penalty = penalties["accuracy_penalty_rate"] * accuracy_error
#             reward += accuracy_penalty
            
#             # 4. Change penalties
#             if new_Ta != Ta
#                 change_magnitude = abs(new_Ta - Ta)
#                 time_to_announced = max(1, Ta - t)
                
#                 # Base penalty for any change
#                 base_penalty = -penalties["base_change_penalty"]
                
#                 # # Magnitude penalty  
#                 # magnitude_penalty = -penalties["magnitude_penalty_rate"] * change_magnitude
                
#                 # # Timing penalty
#                 if time_to_announced == 1
#                     # Day before: high penalty mostly independent of magnitude
#                     timing_penalty = -penalties["step_before_penalty"]
#                 else
#                     # General case: penalty scales with time to deadline/accounced time
#                     urgency_factor = 1.0 / time_to_announced
#                     timing_penalty = -penalties["timing_penalty_rate"] * urgency_factor * (1.0 + penalties["magnitude_scaling"] * change_magnitude)
#                 end
                
#                 # # Direction bias
#                 # direction_penalty = 0.0
#                 # if new_Ta < Tt
#                 #     direction_penalty += -penalties["early_bias_penalty"] * change_magnitude / time_to_announced
#                 # elseif new_Ta > Tt
#                 #     direction_penalty = -penalties["late_bias_penalty"] * change_magnitude / time_to_announced
#                 # end
                
#                 # reward += base_penalty + magnitude_penalty + timing_penalty + direction_penalty
#             end
            
#             return reward
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

# Debugging Version with simple observation functions

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

#             # Debug Step 1: If we have deterministic observation that provide the true end time, does the action announce
#             # the true end time.
#             # Answer: Yes, immediately after the first observation the announce time will be the true end time and stay there
#             # return Deterministic((t, Ta, Tt))

#             # Debug Step 2: We try the same thing but with just a single SparseCat observation
#             # This should be the same as the deterministic observation.
#             # Answer: Yes, this works as expected and gives the same results as above
#             # return SparseCat([(t, Ta, Tt)], [1.0])

#             # Debug Step 3: Now we try with a simple, but heavily weighted observation function
#             # This funciton allows for observation of the true end time and potentially other times 
#             # one timestep before or after the true end time. The true end time is heavily weighted
#             # as the most likely observation.
#             # 
#             # Result: 
#             # This can be solved with CXX (BASE) SARSOP, but Native SARSOP fails to come up with good solutions
#             # all policies seem to have a bias with Native SARSOP

#             # obs_list = [(t, Ta, Tt)]

#             # # Account for the fact that the true end time might be at the minimum or maximum end time
#             # # And so we don't add observations that are out of bounds
#             # if Tt > min_end_time
#             #     obs_list = push!(obs_list, (t, Ta, Tt - 1))
#             # end

#             # if Tt < max_end_time
#             #     obs_list = push!(obs_list, (t, Ta, Tt + 1))
#             # end

#             # # Define probabilities for the observations
#             # if length(obs_list) == 1
#             #     probs = [1.0]
#             # elseif length(obs_list) == 2
#             #     # If we have two observations, we can assign equal probabilities
#             #     probs = [0.8, 0.2]
#             # else
#             #     # If we have three observations, assign equal probabilities
#             #     probs = [0.6, 0.2, 0.2]
#             # end

#             # return SparseCat(obs_list, probs)

#         end,

#         reward = function(s, a)
#             # Reward for taking action a in state s
#             t, Ta, Tt = s

#             # Dead-simple reward - just penalize for the difference between announced and true end time
#             r = -1 * abs(Ta - Tt)

#             # This was the original reward logic of penalizing based on the action (announced time) vs the true end time
#             # but it has some strange behavior
#             # r = -1 * abs(a.announced_time - Tt)

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

# With new observation uncertainty + simple reward function

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

#             # Number of time steps until the true end time
#             remaining = max(Tt - t, 0)

#             if remaining == 0
#                 # If the project is done, we can only observe the true end time
#                 return Deterministic((t, Ta, Tt))
#             end

#             # ------------------------------------------------------------------
#             # 1. Choose an integer “σ” that controls the half-width of the window
#             #    σ  =  ceil( remaining / τ ),  but never smaller than 1
#             #    – large τ  →  σ shrinks *quickly*  (observations lock on sooner)
#             #    – small τ  →  σ shrinks *slowly*
#             # ------------------------------------------------------------------
#             τ = 2  # This is a hyperparameter that can be adjusted
#             σ = remaining / τ

#             # 2. Generate the candidate observed end-times  (Tt-3σ … Tt+3σ)
#             lo = max(min_end_time, Tt - 3σ, t + 1)
#             hi = min(max_end_time, Tt + 3σ, Tt + remaining)

#             # 3. Discrete Gaussian weights, truncated to keep the list sparse
#             obs_list = Tuple{Int,Int,Int}[]
#             w        = Float64[]
#             σ² = (σ^2)
#             for To in lo:hi
#                 push!(obs_list, (t, Ta, To))
#                 push!(w, exp(-((To - Tt)^2) / (2σ²)))
#             end
#             p = w ./ sum(w) # normalise

#             return SparseCat(obs_list, p)
#         end,

#         reward = function(s, a)
#             # Reward for taking action a in state s
#             t, Ta, Tt = s
            
#             # Dead-simple reward - just penalize for the difference between announced and true end time
#             r = -1 * abs(Ta - Tt)

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

# Original w/ simplified reward function

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

            # If the project is done or we are at the timestep before the maximum project completion time
            # just return Tt deterministically
            if t >= Tt || t + 1 == max_end_time
                return Deterministic((t, Ta, Tt))
            end

            # Otherwise, we have the case where the project is not done yet
            # The minimum completion time we can observe the maximium of 
            # the current time plus 1 (i.e. we think that after the next transition we will be done)
            # and the minimum observed completion time (min_end_time)
            min_obs_time = max(t + 1, min_end_time)

            possible_Tos = collect(min_obs_time:max_end_time)

            # Calculate parameters of the truncated normal
            mu = Tt
            std = (Tt - t) / 1.5
            
            if Tt - t <= 0 
                # The task is done, so we should observe the true end time
                return Deterministic((t, Ta, Tt))
            end

            base_dist = Normal(mu, std)
            # the truncated normal distribution is defined from t+1 to max_end_time
            lower = min_obs_time
            upper = max_end_time
            cdf_lower = cdf(base_dist, lower)
            cdf_upper = cdf(base_dist, upper)
            denom = cdf_upper - cdf_lower

            probs = Float64[]
            for To_val in possible_Tos
                p = (pdf(base_dist, To_val) / denom)
                push!(probs, p)
            end

            total_p = sum(probs)
            if total_p == 0.0
                return Deterministic((t, Ta, Tt))
            end
            probs ./= total_p

            obs_list = [(t, Ta, To_val) for To_val in possible_Tos]
            
            return SparseCat(obs_list, probs)
        end,

        reward = function(s, a)
            # Reward for taking action a in state s
            t, Ta, Tt = s
            
            # Dead-simple reward - just penalize for the difference between announced and true end time
            r = -1 * abs(Ta - Tt)

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

# Original

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