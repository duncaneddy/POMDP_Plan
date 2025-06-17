"""
Debugging function to plot the distribution of observations used by 
Emma and Niles at different time points over a simulation.
"""

using Distributions           # Normal(), pdf(), cdf()
using PlotlyJS                # bar(), Plot, Layout, etc.

"""
    plot_obs_hist_truncnorm(time_points;
                            Tt, min_end_time, max_end_time,
                            Ta=nothing, std_div=1.5,
                            title="P(To | t) – truncated normal")

Create a grouped-bar chart of the observation distribution P(To | t) for each
`t ∈ time_points`, following the **truncated normal** rule you provided:

* If the task is done (`t ≥ Tt`) or only one step remains, a single bar at
  `To = Tt` has probability 1.
* Otherwise  
  σ   = (Tt − t) / std_div  
  lower = max(t+1, min_end_time)  
  upper = max_end_time  
  p(To) ∝  pdf(Normal(μ=Tt, σ), To) ,  then renormalised over the integer
  grid `lower:upper`.

Keyword `std_div` is the 1.5 in your code (feel free to change it).
Returns a `PlotlyJS.Plot`, so call `display(fig)` or `savefig(fig, "out.html")`.
"""
function plot_obs_hist_truncnorm(time_points::Vector{<:Integer};
                                 Tt::Integer,
                                 min_end_time::Integer,
                                 max_end_time::Integer,
                                 Ta=nothing,
                                 std_div::Real = 1.5,
                                 title::String = "Observation probability P(To | t) (truncated normal)")

    # ------------ global x-axis so bars stay aligned ----------------------
    global_min_To = max(minimum(time_points) + 1, min_end_time)
    x_vals = collect(global_min_To:max_end_time)

    traces = PlotlyJS.GenericTrace[]

    for t in time_points
        y = zeros(Float64, length(x_vals))

        # ---------- deterministic edge cases ----------
        if (t ≥ Tt) || (t + 1 == max_end_time) || (Tt - t ≤ 0)
            if Tt in x_vals
                y[findfirst(==(Tt), x_vals)] = 1.0
            end
        else
            # ---------- truncated normal over integer grid ----------
            lower = max(t + 1, min_end_time)
            upper = max_end_time
            μ     = Tt
            σ     = (Tt - t) / std_div
            base  = Normal(μ, σ)

            denom = cdf(base, upper) - cdf(base, lower)
            if denom > 0
                probs_local = [pdf(base, To)/denom for To in lower:upper]
                s           = sum(probs_local)
                if s > 0
                    probs_local ./= s             # final normalisation
                    # map onto global x positions
                    for (To, p) in zip(lower:upper, probs_local)
                        y[findfirst(==(To), x_vals)] = p
                    end
                end
            end
        end

        push!(traces,
              bar(x=x_vals,
                  y=y,
                  name="t = $t",
                  offsetgroup=string(t)))
    end

    layout = Layout(barmode="group",
                    bargap=0.15,
                    bargroupgap=0.05,
                    xaxis_title="Observed completion time  Tₒ",
                    yaxis_title="Probability",
                    title=title,
                    template="simple_white")

    return Plot(traces, layout)
end

fig = plot_obs_hist_truncnorm(collect(1:15);  # time points to plot
                    Tt=15,
                    min_end_time=10,
                    max_end_time=20,
                    std_div=1.5)
savefig(fig, "obs_hist_old.png")