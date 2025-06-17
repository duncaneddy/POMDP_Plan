"""
Simple debugging function to plot distribution of observations at different time
points over a simulation.
"""

using Distributions          # only for SparseCat type-hint in comments
using PlotlyJS

"""
    plot_obs_hist(time_points; Tt, min_end_time, max_end_time,
                  Ta=nothing, τ=2, title="P(To | t) with discrete Gaussian window")

Generate a grouped-bar PlotlyJS figure showing the observation distribution
for each `t` in `time_points`, using the *discrete Gaussian* logic:

σ  = max( ceil(remaining / τ), 1 )
lo = max(min_end_time, Tt - 3σ, t)
hi = min(max_end_time, Tt + 3σ, Tt + remaining)
w  ∝ exp(-(To − Tt)² / (2σ²))

Edge case `remaining == 0` collapses to a single deterministic bar at `To = Tt`.
"""
function plot_obs_hist(time_points::Vector{<:Integer};
                       Tt::Integer,
                       min_end_time::Integer,
                       max_end_time::Integer,
                       Ta=nothing,          # placeholder for signature parity
                       τ::Real = 2,
                       title::String = "Observation probability P(To | t)")

    # ---------- helper that returns (x_vals, p_vec) ------------------------
    discrete_gaussian_probs(t) = begin
        remaining = max(Tt - t, 0)
        # deterministic if done
        if remaining == 0
            return ([Tt], [1.0])
        end

        σ = max(ceil(Int, remaining / τ), 1)
        lo = max(min_end_time, Tt - 3σ, t + 1)
        hi = min(max_end_time, Tt + 3σ, Tt + remaining)

        xs = collect(lo:hi)
        σ2 = σ^2
        w  = [exp(-((To - Tt)^2) / (2σ2)) for To in xs]
        p  = w ./ sum(w)
        return (xs, p)
    end

    # Build a *global* x-axis to keep bars aligned
    all_To_vals = unique!(vcat([discrete_gaussian_probs(t)[1] for t in time_points]...))
    sort!(all_To_vals)

    traces = PlotlyJS.GenericTrace[]
    for t in time_points
        xs, p = discrete_gaussian_probs(t)
        # map probs onto the global x-axis
        y = zeros(Float64, length(all_To_vals))
        for (To, prob) in zip(xs, p)
            y[findfirst(==(To), all_To_vals)] = prob
        end
        push!(traces,
              bar(x=all_To_vals,
                  y=y,
                  name="t = $t",
                  offsetgroup=string(t)))
    end

    layout = Layout(barmode="group",
                    bargap=0.15,
                    bargroupgap=0.05,
                    xaxis_title="Observed completion time To",
                    yaxis_title="Probability",
                    title=title,
                    template="simple_white")

    return Plot(traces, layout)
end

fig = plot_obs_hist(collect(1:15);  # time points to plot
                    Tt=15,
                    min_end_time=10,
                    max_end_time=20,
                    τ=2)          # adjust τ to tighten / loosen the window
display(fig)       # or savefig(fig, "obs_hist.html")

savefig(fig, "obs_hist.png")