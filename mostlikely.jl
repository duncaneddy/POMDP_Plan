# This is a policy where the agent always announces the time with the highest belief probability 
# if that is different from the current announced time

struct MostLikelyPolicy <: Policy end

function POMDPs.action(policy::MostLikelyPolicy, belief, state=nothing)
    # Extract the most likely state from the belief
    states = belief.state_list
    probs = belief.b
    max_prob_index = argmax(probs)
    # s is the most likely state
    s = states[max_prob_index]

    # Unpack state variables
    t, Ta, Ts = s

    # Ts is the most likely true end time
    observed_time = Ts

    # Determine the action
    if observed_time != Ta
        return AnnounceAction(observed_time)
    else
        return :dont_announce
    end
end

