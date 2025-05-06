function create_evaluation_plots(stats, output_dir)
    # Make sure plots directory exists
    plots_dir = joinpath(output_dir, "plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Plot reward distribution
    p1 = histogram(stats["rewards"], 
                 bins=20, 
                 title="Reward Distribution",
                 xlabel="Total Reward",
                 ylabel="Frequency",
                 legend=false,
                 fillalpha=0.7,
                 color=:blue)
    
    # Add mean and median lines
    reward_mean = mean(stats["rewards"])
    reward_median = median(stats["rewards"])
    vline!([reward_mean], label="Mean", linewidth=2, color=:red)
    vline!([reward_median], label="Median", linewidth=2, color=:green, linestyle=:dash)
    
    savefig(p1, joinpath(plots_dir, "reward_distribution.png"))
    
    # Plot error metrics
    p2 = plot(title="Error Metrics",
            xlabel="Metric",
            ylabel="Value",
            legend=false,
            xticks=(1:3, ["Initial Error", "Final Error", "Change Magnitude"]),
            grid=false,
            boxplot=true)
    
    boxplot!(p2, [1], stats["initial_errors"], fillalpha=0.7, color=:blue)
    boxplot!(p2, [2], stats["final_errors"], fillalpha=0.7, color=:red)
    boxplot!(p2, [3], stats["avg_change_magnitudes"], fillalpha=0.7, color=:green)
    
    savefig(p2, joinpath(plots_dir, "error_metrics.png"))
    
    # Plot number of changes
    p3 = histogram(stats["num_changes"],
                 bins=maximum(stats["num_changes"]) - minimum(stats["num_changes"]) + 1,
                 title="Number of Announcement Changes",
                 xlabel="Number of Changes",
                 ylabel="Frequency",
                 legend=false,
                 fillalpha=0.7,
                 color=:purple)
    
    savefig(p3, joinpath(plots_dir, "num_changes.png"))
    
    # Plot undershoot vs overshoot frequency
    undershoot_count = sum(stats["final_undershoot"])
    overshoot_count = length(stats["final_undershoot"]) - undershoot_count
    
    p4 = pie(["Undershoot", "Overshoot"], 
           [undershoot_count, overshoot_count],
           title="Final Announcement Type",
           legend=false,
           colors=[:blue, :orange],
           annotations=(1:2, [text("$undershoot_count\n($(round(100*undershoot_count/length(stats["final_undershoot"]), digits=1))%)", 8),
                           text("$overshoot_count\n($(round(100*overshoot_count/length(stats["final_undershoot"]), digits=1))%)", 8)]))
    
    savefig(p4, joinpath(plots_dir, "undershoot_overshoot.png"))
    
    return [p1, p2, p3, p4]
end