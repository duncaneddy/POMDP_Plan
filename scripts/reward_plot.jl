###############################################################################
# Enhanced Reward-surface visualizer for deadline-management POMDP
#
#   x-axis  = Δt      = days remaining until the *true* completion time T_true
#   y-axis  = |ΔTa|   = magnitude of the change you make to the announced date
#
#   Colour  = total penalty incurred *at that moment*, using the enhanced
#             reward structure with configurable management styles
#
# Features the enhanced penalty structure with:
#   - Base change penalty (always applied)
#   - Magnitude scaling (larger changes cost more)  
#   - Timing urgency (penalties increase as deadline approaches)
#   - Day-before rule (special high penalty mostly independent of magnitude)
#   - Direction bias (configurable preference for early vs late announcements)
#
# Requires: ] add PlotlyJS ColorSchemes
###############################################################################

using PlotlyJS, ColorSchemes

# ─────────────────────────────── parameters ───────────────────────────────── #
const MIN_TIME   =  1          # earliest plotted Δt  (≥1 keeps denom safe)
const MAX_TIME   = 20          # latest plotted Δt
const MAX_CHANGE = 15          # maximum change magnitude to plot

# convenience
Δt_vals  = MIN_TIME:MAX_TIME                   # x-axis (days remaining)
ΔTa_vals = 0:MAX_CHANGE                        # y-axis (|change| magnitude)

# ═══════════════════════════════════════════════════════════════════════════ #
#                           ENHANCED PENALTY STRUCTURE                         #
# ═══════════════════════════════════════════════════════════════════════════ #

# Penalty configuration - easily adjustable for different management styles
mutable struct PenaltyConfig
    # Base penalties
    base_change_penalty::Float64        # Always applied for any change
    magnitude_penalty_rate::Float64     # Per unit of change magnitude
    
    # Timing penalties
    timing_penalty_rate::Float64        # Base urgency penalty coefficient  
    magnitude_scaling::Float64          # How much magnitude affects timing penalty
    
    # Day-before special case
    day_before_penalty::Float64         # High penalty for day-before changes
    magnitude_independence_factor::Float64  # Small magnitude scaling for day-before
    
    # Direction bias (assumes we know if announcing early/late relative to true end)
    early_bias_penalty::Float64         # Additional penalty for early announcements
    late_bias_penalty::Float64          # Additional penalty for late announcements
    direction_bias_enabled::Bool        # Whether to include direction bias
end

# Predefined management style configurations
function conservative_style()
    return PenaltyConfig(
        8.0,   # base_change_penalty
        3.0,   # magnitude_penalty_rate
        25.0,  # timing_penalty_rate
        0.6,   # magnitude_scaling
        60.0,  # day_before_penalty
        2.0,   # magnitude_independence_factor
        10.0,  # early_bias_penalty
        15.0,  # late_bias_penalty
        false  # direction_bias_enabled (for cleaner visualization)
    )
end

function balanced_style()
    return PenaltyConfig(
        5.0,   # base_change_penalty
        2.0,   # magnitude_penalty_rate
        20.0,  # timing_penalty_rate
        0.5,   # magnitude_scaling
        40.0,  # day_before_penalty
        3.0,   # magnitude_independence_factor
        8.0,   # early_bias_penalty
        12.0,  # late_bias_penalty
        false  # direction_bias_enabled
    )
end

function agile_style()
    return PenaltyConfig(
        3.0,   # base_change_penalty
        1.5,   # magnitude_penalty_rate
        15.0,  # timing_penalty_rate
        0.4,   # magnitude_scaling
        25.0,  # day_before_penalty
        2.5,   # magnitude_independence_factor
        6.0,   # early_bias_penalty
        9.0,   # late_bias_penalty
        false  # direction_bias_enabled
    )
end

# ─────────────────────────── penalty calculation ─────────────────────────── #

function enhanced_change_penalty(ΔTa::Float64, Δt::Float64, config::PenaltyConfig)
    """
    Calculate total penalty for making a change of magnitude ΔTa with Δt days remaining.
    
    Args:
        ΔTa: Magnitude of announcement change (always ≥ 0)
        Δt: Days remaining until true end time (always ≥ 1)
        config: Penalty configuration struct
    
    Returns:
        Total penalty (negative value)
    """
    
    penalty = 0.0
    
    # ═══ COMPONENT 1: Base penalty ═══
    # Always some cost for making any change (except when ΔTa = 0)
    if ΔTa > 0
        penalty += config.base_change_penalty
    end
    
    # ═══ COMPONENT 2: Magnitude penalty ═══  
    # Larger changes cost more
    penalty += config.magnitude_penalty_rate * ΔTa
    
    # ═══ COMPONENT 3: Timing penalty ═══
    # Changes closer to deadline cost much more
    if ΔTa > 0  # Only apply if there's actually a change
        if Δt == 1
            # Special case: day before true end time
            # High penalty that's nearly independent of magnitude
            timing_penalty = config.day_before_penalty + config.magnitude_independence_factor * ΔTa
            penalty += timing_penalty
        else
            # General case: penalty inversely related to time remaining
            urgency_factor = 1.0 / Δt
            timing_penalty = config.timing_penalty_rate * urgency_factor * (1.0 + config.magnitude_scaling * ΔTa)
            penalty += timing_penalty
        end
    end
    
    # ═══ COMPONENT 4: Direction bias ═══
    # Different penalties for announcing earlier vs later than true end time
    # (This component is optional and can be disabled for cleaner visualization)
    if config.direction_bias_enabled && ΔTa > 0
        # For visualization purposes, assume we're making the change in the "worse" direction
        # In practice, this would depend on whether new_announced < true_end_time
        direction_penalty = max(config.early_bias_penalty, config.late_bias_penalty) * ΔTa / Δt
        penalty += direction_penalty
    end
    
    return -penalty  # Return as negative (penalty)
end

# ─────────────────────────── plotting functions ─────────────────────────── #

function create_penalty_surface(config::PenaltyConfig, title_suffix::String="")
    """Create penalty surface visualization for given configuration."""
    
    # Build Z matrix: rows = ΔTa values, columns = Δt values
    Z = [enhanced_change_penalty(Float64(ΔTa), Float64(Δt), config) for ΔTa in ΔTa_vals, Δt in Δt_vals]
    
    title_text = "Enhanced Change Penalty Surface" * (title_suffix != "" ? " - $title_suffix" : "")
    
    plt = Plot(
        heatmap(
            x = Δt_vals,
            y = ΔTa_vals,
            z = Z,
            colorscale = "Viridis",  # Good perceptual colorscale
            reversescale = true,     # More negative = darker (more penalty)
            colorbar = attr(title = "Penalty"),
            hovertemplate = "Days remaining: %{x}<br>Change magnitude: %{y}<br>Penalty: %{z:.1f}<extra></extra>"
        ),
        Layout(
            title = attr(text = title_text, x = 0.5),
            xaxis = attr(title = "Days remaining to true end (Δt)"),
            yaxis = attr(title = "Magnitude of change |ΔTa| (days)"),
            margin = attr(l = 80, r = 30, t = 70, b = 70),
            width = 800,
            height = 600
        )
    )
    
    return plt, Z
end

function create_comparison_plot()
    """Create side-by-side comparison of different management styles."""
    
    configs = [
        (conservative_style(), "Conservative"),
        (balanced_style(), "Balanced"), 
        (agile_style(), "Agile")
    ]
    
    plots = []
    
    for (config, style_name) in configs
        Z = [enhanced_change_penalty(Float64(ΔTa), Float64(Δt), config) for ΔTa in ΔTa_vals, Δt in Δt_vals]
        
        plt = Plot(
            heatmap(
                x = Δt_vals,
                y = ΔTa_vals,
                z = Z,
                colorscale = "Viridis",
                reversescale = true,
                showscale = false,  # Only show colorbar on last plot
                hovertemplate = "Δt: %{x}<br>|ΔTa|: %{y}<br>Penalty: %{z:.1f}<extra></extra>"
            ),
            Layout(
                title = attr(text = style_name, x = 0.5, font_size = 14),
                xaxis = attr(title = "Days remaining (Δt)"),
                yaxis = attr(title = "Change magnitude |ΔTa|"),
                margin = attr(l = 60, r = 10, t = 50, b = 50),
                width = 300,
                height = 400
            )
        )
        push!(plots, plt)
    end
    
    # Add colorbar to the last plot
    plots[end].data[1].showscale = true
    plots[end].data[1].colorbar = attr(title = "Penalty")
    
    return plots
end

function analyze_penalty_structure(config::PenaltyConfig)
    """Print analysis of penalty structure for given configuration."""
    
    println("\n" * "="^70)
    println("PENALTY STRUCTURE ANALYSIS")
    println("="^70)
    
    # Test specific scenarios
    scenarios = [
        (1.0, 1.0, "Small change, day before deadline"),
        (5.0, 1.0, "Large change, day before deadline"),
        (1.0, 5.0, "Small change, 5 days before"),
        (5.0, 5.0, "Large change, 5 days before"),
        (1.0, 15.0, "Small change, 15 days before"),
        (5.0, 15.0, "Large change, 15 days before"),
        (0.0, 5.0, "No change (baseline)")
    ]
    
    for (ΔTa, Δt, description) in scenarios
        penalty = enhanced_change_penalty(ΔTa, Δt, config)
        println("$description: Penalty = $(round(penalty, digits=2))")
    end
    
    println("\nKey ratios:")
    day_before_small = enhanced_change_penalty(1.0, 1.0, config)
    day_before_large = enhanced_change_penalty(5.0, 1.0, config)
    early_small = enhanced_change_penalty(1.0, 10.0, config)
    early_large = enhanced_change_penalty(5.0, 10.0, config)
    
    println("Day-before magnitude sensitivity: $(round(day_before_large / day_before_small, digits=2))x")
    println("Early-phase magnitude sensitivity: $(round(early_large / early_small, digits=2))x")
    println("Timing effect (small changes): $(round(early_small / day_before_small, digits=2))x")
    println("Timing effect (large changes): $(round(early_large / day_before_large, digits=2))x")
end

# ═══════════════════════════════════════════════════════════════════════════ #
#                                    MAIN                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

function main()
    println("Enhanced Deadline Management POMDP - Reward Surface Visualizer")
    println("=" ^ 65)
    
    # Choose configuration (easily changeable)
    config = balanced_style()  # Change this to conservative_style() or agile_style()
    
    # Analyze the penalty structure
    analyze_penalty_structure(config)
    
    # Create main visualization
    println("\nGenerating penalty surface visualization...")
    plt, Z = create_penalty_surface(config, "Balanced Management Style")
    
    # Save the plot
    savefig(plt, "enhanced_reward_surface.html")
    println("Main plot saved as: enhanced_reward_surface.html")
    
    # Create comparison visualization
    println("\nGenerating management style comparison...")
    comparison_plots = create_comparison_plot()
    
    # Save comparison plots individually
    styles = ["Conservative", "Balanced", "Agile"]
    for (i, (plt, style)) in enumerate(zip(comparison_plots, styles))
        filename = "penalty_surface_$(lowercase(style)).html"
        savefig(plt, filename)
        println("$style style plot saved as: $filename")
    end
    
    # Print some statistics
    println("\n" * "─"^50)
    println("SURFACE STATISTICS")
    println("─"^50)
    println("Penalty range: $(round(minimum(Z), digits=1)) to $(round(maximum(Z), digits=1))")
    println("Maximum penalty ratio: $(round(maximum(Z) / minimum(Z), digits=1))x")
    
    # Find the steepest gradient (most sensitive region)
    max_gradient = 0.0
    max_gradient_point = (0, 0)
    
    for i in 1:(length(ΔTa_vals)-1), j in 1:(length(Δt_vals)-1)
        gradient = abs(Z[i+1, j] - Z[i, j]) + abs(Z[i, j+1] - Z[i, j])
        if gradient > max_gradient
            max_gradient = gradient
            max_gradient_point = (ΔTa_vals[i], Δt_vals[j])
        end
    end
    
    println("Steepest gradient at: ΔTa=$(max_gradient_point[1]), Δt=$(max_gradient_point[2])")
    println("This represents the most change-sensitive region of the penalty landscape.")
    
    return plt, Z
end

# Run the visualization
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end