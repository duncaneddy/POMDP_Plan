
function extract_belief_states_and_probs(belief)
    if isa(belief, POMDPTools.POMDPDistributions.SparseCat)
        # For SparseCat, we can directly access the states and probabilities
        return belief.vals, belief.probs
    elseif isa(belief, Tuple{Int64, Int64, Int64})
        # If belief is a tuple, it is a single state with probability 1
        return [belief], [1.0]
    else
        states = belief.state_list
        probs = belief.b
        return states, probs
    end
end


function highest_belief_state(belief)
    # If the belief is a sparseCat we handle it differently
    if isa(belief, POMDPTools.POMDPDistributions.SparseCat)
        # Return middle index of belief.vals
        return belief.vals[ceil(Int, length(belief.vals) / 2)]
    elseif isa(belief, Tuple{Int64, Int64, Int64})
        # If belief is a tuple, it is a single state
        return belief
    else
        states, probs = extract_belief_states_and_probs(belief)
        max_prob = maximum(probs)
        max_prob_index = argmax(probs)
        return states[max_prob_index]
    end
end


function print_belief_states_and_probs(belief)
    states, probs = extract_belief_states_and_probs(belief)
    for (state, prob) in zip(states, probs)
        if prob > 0
            println("State: $state, Probability: $prob")
        end
    end
end