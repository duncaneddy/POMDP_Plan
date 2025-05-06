
function extract_belief_states_and_probs(belief)
    states = belief.state_list
    probs = belief.b
    return states, probs
end


function highest_belief_state(belief)
    states, probs = extract_belief_states_and_probs(belief)
    max_prob = maximum(probs)
    max_prob_index = argmax(probs)
    return states[max_prob_index]
end


function print_belief_states_and_probs(belief)
    states, probs = extract_belief_states_and_probs(belief)
    for (state, prob) in zip(states, probs)
        if prob > 0
            println("State: $state, Probability: $prob")
        end
    end
end