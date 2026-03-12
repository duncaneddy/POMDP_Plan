#!/usr/bin/env julia
# Regenerate pareto plots from existing results JSON without rerunning simulations
# Also re-evaluates baseline solvers (OBSERVEDTIME, MOSTLIKELY) which are fast heuristics

using Pkg
push!(LOAD_PATH, dirname(dirname(@__FILE__)))

using JSON
using Random
using Statistics
include(joinpath(@__DIR__, "pareto_frontier_analysis.jl"))

results_dir = joinpath(dirname(@__DIR__), "final_paper_results", "pareto_analysis_results_final")
results_path = joinpath(results_dir, "pareto_sweep_results.json")

println("Loading results from: $results_path")
results = JSON.parsefile(results_path)

# Convert error_metrics from Dict to NamedTuple so getfield() works
for r in results
    if haskey(r, "error_metrics") && isa(r["error_metrics"], Dict)
        em = r["error_metrics"]
        r["error_metrics"] = (;
            avg_weighted_error = em["avg_weighted_error"],
            max_error = em["max_error"],
            final_error = em["final_error"],
            avg_absolute_error = em["avg_absolute_error"],
            avg_timesteps_with_error = em["avg_timesteps_with_error"],
            rms_error = em["rms_error"],
        )
    end
end

println("Loaded $(length(results)) sweep results")

# Re-evaluate baseline solvers to add reference points
println("Evaluating baseline solvers (OBSERVEDTIME, MOSTLIKELY)...")
Random.seed!(42)

min_end_time = DEFAULT_MIN_END_TIME
max_end_time = DEFAULT_MAX_END_TIME
num_conditions = DEFAULT_NUM_CONDITIONS
num_repetitions = DEFAULT_NUM_REPETITIONS

initial_conditions_raw = POMDPPlanning.generate_initial_conditions(
    min_end_time, max_end_time, num_conditions, seed=42)

initial_conditions = []
for ic in initial_conditions_raw
    for _ in 1:num_repetitions
        push!(initial_conditions, (0, ic["Ta"], ic["Tt"]))
    end
end

baseline_results = evaluate_baseline_solvers(
    min_end_time, max_end_time, initial_conditions,
    ["OBSERVEDTIME", "MOSTLIKELY"],
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_DISCOUNT, DEFAULT_STD_DIVISOR,
    true,
    policy_timeout=DEFAULT_POLICY_TIMEOUT
)

println("Baseline evaluation complete. Adding $(length(baseline_results)) baseline results.")

# Combine sweep results with baselines
combined_results = vcat(results, baseline_results)

create_pareto_plots(combined_results, results_dir)
println("Done! Plots regenerated in: $results_dir/plots/")
