#!/usr/bin/env julia

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

# Load the module and run the main function
using POMDPPlanning
exit(POMDPPlanning.main())