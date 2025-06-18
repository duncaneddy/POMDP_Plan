#!/usr/bin/env julia

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

# Load the module and run the main function
using POMDPPlanning
using SARSOP
using POMDPs
using QuickPOMDPs
using POMDPTools
using JLD2

# Define pomdp
# CHANGE THIS TO MATCH YOUR POMDP
min_end_time = 10
max_end_time = 20
discount_factor = 0.9999

# Change to match
pomdp = POMDPPlanning.define_pomdp(
    min_end_time,
    max_end_time,
    discount_factor,
    verbose=true
)

# Define input and output num_states
input_policy = "policy_116_326.73" # Change tthis 

policy = SARSOP.load_policy(pomdp, "$input_policy.policy")

policy_filepath = joinpath(".", "$input_policy.jld2")
save(policy_filepath, "policy", policy)