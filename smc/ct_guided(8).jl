wd = @__DIR__
cd(wd)

"""
Continuous-time guided proposals with ε-regularisation.

SDE:    dX_t = b(X_t) dt + q dW_t,   X_0 = x0
Observation: v = X_T + Z,  Z ~ N(0, σ_obs²)

so h(x_T) = φ(v; x_T, σ_obs²).

Linear reference process: dX_t = -κ X_t dt + q dW_t
giving Gaussian g(t,x) = φ(x; μ(t), Σ(t)) via backward ODEs.

Guided proposal (ε-regularised):
  dX_t = b(X_t) dt + Q r_ε(t,X_t) dt + q dW_t
where
  r_ε(t,x) = g(t,x)/(g(t,x)+ε) · ∇_x log g(t,x)
            = -g(t,x)/(g(t,x)+ε) · Σ(t)⁻¹(x - μ(t))

IS weight:
  G_ε(x) = h(x_T) / (g(T, x_T) + ε) · (g(x_0) + ε) · Ψ(x)
where Ψ is the Girsanov correction between P and P^{g+ε}.
"""

using Distributions
using Random
using Statistics
using Plots
using Parameters
using UnPack

mkpath("figs")

# ── Parameters ────────────────────────────────────────────────────────────────

@with_kw struct ParaCT{R}
    # SDE parameters
    x0::R    = 0.0       # initial condition
    κ::R     = 1.0       # linear damping (reference process)
    α::R     = 0.2       # nonlinear oscillation amplitude
    q::R     = 3.0       # diffusion coefficient
    # Observation
    v::R     = 1.0       # observed value
    σ_obs::R = 1.0       # observation noise std
    # Time grid
    T::R     = 1.0       # terminal time
    n_steps::Int = 2500  # number of Euler-Maruyama steps
end

# ── True drift (nonlinear) ────────────────────────────────────────────────────

"""b(x) = -κx + α sin(x)  (dissipative nonlinear drift)"""
b(x, p) = -p.κ * x + p.α * sin(x)

"""Linear reference drift: b̃(x) = -κx"""
b_ref(x, p) = -p.κ * x

# ── Backward ODEs for Gaussian g(t,x) = φ(x; μ(t), Σ(t)) ────────────────────
# Reference process: dX = -κX dt + q dW
# Terminal condition: g(T,x) = h(x_T) = φ(v; x_T, σ_obs²) = φ(x_T; v, σ_obs²)
# so μ(T) = v, Σ(T) = σ_obs²
#
# Backward ODEs for reference process dX = -κX dt + q dW:
#   ∂_t g + (-κx)∂_x g + (q²/2)∂_xx g = 0
# Substituting Gaussian ansatz g = N(x; μ(t), Σ(t)):
#   μ̇ = κμ        →  μ(t) = v · exp(-κ(T-t))
#   Σ̇ = 2κΣ - q²  →  Σ(t) = (σ_obs² - q²/(2κ)) exp(2κ(T-t)) + q²/(2κ)
# Note: Σ(t) > 0 for all t < T since the exp term dominates.

# function gaussian_g_params(t, p)
#     @unpack κ, q, v, σ_obs, T = p
#     μt = v * exp(-κ * (T - t))
#     Σt = (σ_obs^2 - q^2/(2κ)) * exp(2κ*(T-t)) + q^2/(2κ)
#     Σt = max(Σt, 1e-10)   # numerical safety
#     return μt, Σt
# end

function gaussian_g_params(t, p)
    @unpack κ, q, v, σ_obs, T = p
    μt = v * exp(-κ * (T - t))
    Σt = (σ_obs^2 + q^2/(2κ)) * exp(2κ*(T-t)) - q^2/(2κ)
    return μt, Σt
end

"""g(t, x, p) = φ(x; μ(t), Σ(t))"""
function g_ct(t, x, p)
    μt, Σt = gaussian_g_params(t, p)
    return pdf(Normal(μt, sqrt(Σt)), x)
end

"""∇_x log g(t, x, p) = -(x - μ(t)) / Σ(t)"""
function score_g(t, x, p)
    μt, Σt = gaussian_g_params(t, p)
    return -(x - μt) / Σt
end

"""r_ε(t, x, p) = g(t,x)/(g(t,x)+ε) · ∇_x log g(t,x)"""
function r_eps(t, x, ε, p)
    gval = g_ct(t, x, p)
    return gval / (gval + ε) * score_g(t, x, p)
end

# ── Observation density ───────────────────────────────────────────────────────

"""h(x_T) = φ(v; x_T, σ_obs²)"""
h_ct(xT, p) = pdf(Normal(xT, p.σ_obs), p.v)

# ── Euler-Maruyama simulation ─────────────────────────────────────────────────

"""
Simulate one path of the guided proposal P^{g+ε} via Euler-Maruyama.
Stores the full path xs and accumulates log Ψ(x) along the path.

log Ψ(x) = ∫ (b(x_t) - b̃(x_t)) r_ε(t,x_t) dt
where b̃ = -κx is the linear reference drift, b - b̃ = α sin(x).

Returns: (xT, log_psi, xs). Accepts either an rng or pre-drawn dWs
(for reuse of the same Brownian path across ε values).
"""
function simulate_guided(ε, p; rng=nothing, dWs=nothing)
    @unpack x0, q, T, n_steps = p
    dt   = T / n_steps
    sqdt = sqrt(dt)
    Q    = q^2

    xs      = Vector{Float64}(undef, n_steps + 1)
    xs[1]   = x0
    log_psi = 0.0
    ts      = range(0.0, T, length=n_steps+1)

    for i in 1:n_steps
        t  = ts[i]
        x  = xs[i]
        dW = isnothing(dWs) ? randn(rng) * sqdt : dWs[i]

        rval       = r_eps(t, x, ε, p)
        drift_guid = b(x, p) + Q * rval
        db         = b(x, p) - b_ref(x, p)
        log_psi   += db * rval * dt

#        if !isfinite(log_psi) || abs(log_psi) > 1e6
if !isfinite(log_psi) || abs(log_psi) > 1e6 || !isfinite(x) || abs(x) > 100
            log_psi = -1e6
            xs[i+1] = x + drift_guid * dt + q * dW
            for j in i+2:n_steps+1; xs[j] = xs[j-1]; end
            break
        end

        xs[i+1] = x + drift_guid * dt + q * dW
    end

    return xs[end], log_psi, xs
end

"""Pre-draw N_steps Brownian increments for reuse across ε values."""
function draw_increments(p, rng)
    @unpack T, n_steps, q = p
    dt   = T / n_steps
    sqdt = sqrt(dt)
    return randn(rng, n_steps) .* sqdt
end

# ── IS weight ─────────────────────────────────────────────────────────────────

"""
IS weight G_ε(x) = h(x_T) · (g(x0)+ε) / (g(T,x_T)+ε) · Ψ(x)

Returns log weight.
"""
function log_weight(xT, log_psi, ε, p)
    g0  = g_ct(0.0, p.x0, p)
    gT  = g_ct(p.T, xT, p)
    hT  = h_ct(xT, p)
    # guard against log(0) when ε=0 and gT≈0
    if hT <= 0.0 || (gT + ε) <= 0.0 || (g0 + ε) <= 0.0
        return -Inf
    end
    return log(hT) + log(g0 + ε) - log(gT + ε) - log_psi
end

# ── m̂(ε) estimator ────────────────────────────────────────────────────────────

"""
Estimate m(ε) = E_{P^{g+ε}}[G_ε(X)²] by simulating N paths.
"""
function estimate_m(ε, N, p, rng)
    weights = Float64[]
    for _ in 1:N
        xT, log_psi, _ = simulate_guided(ε, p; rng=rng)
        lw = log_weight(xT, log_psi, ε, p)
        push!(weights, exp(lw))
    end
    return mean(weights .^ 2) 
end

# ── Unguided simulation (plain SDE, no guiding term) ─────────────────────────

"""
Simulate one path of the unguided process P via Euler-Maruyama.
Stores the full path and Brownian increments so that log Ψ(ε) can be
computed for any ε by replaying the path, without re-simulating.

Returns: (xs, dWs) where xs[i] = x_{t_{i-1}} and dWs[i] = dW_{t_{i-1}}
"""
function simulate_unguided(p, rng)
    @unpack x0, q, T, n_steps = p
    dt   = T / n_steps
    sqdt = sqrt(dt)
    xs   = Vector{Float64}(undef, n_steps + 1)
    dWs  = Vector{Float64}(undef, n_steps)
    xs[1] = x0
    for i in 1:n_steps
        dW      = randn(rng) * sqdt
        dWs[i]  = dW
        xs[i+1] = xs[i] + b(xs[i], p) * dt + q * dW
    end
    return xs, dWs
end

"""
Compute log Ψ(x, ε) along a stored path xs using the deterministic formula
from the notes:

    log Ψ(x) = ∫₀ᵀ (b(x_t) - b̃(x_t)) r_ε(t,x_t) dt

where b̃(x) = -κx is the linear reference drift, so b(x) - b̃(x) = α sin(x).
"""
function log_psi_unguided(xs, ε, p)
    @unpack T, n_steps = p
    dt      = T / n_steps
    log_psi = 0.0
    ts      = range(0.0, T, length=n_steps+1)
    for i in 1:n_steps
        x        = xs[i]
        db       = b(x, p) - b_ref(x, p)   # = α sin(x)
        rval     = r_eps(ts[i], x, ε, p)
        log_psi += db * rval * dt
        if !isfinite(log_psi)
            return -Inf
        end
    end
    return log_psi
end

# ── Find ε* by grid search ────────────────────────────────────────────────────

"""
Estimate m̂(ε) by simulating N_aux guided paths under P^{g+ε} for each ε
in the grid, and computing:

    m̂(ε) = (1/N) Σ_i G_ε(x^i)²
    G_ε(x) = h(xT) · (g0+ε)/(gT+ε) · exp(-log_psi)

where log_psi = log Ψ(x) is accumulated along the guided path.
Reuses the same Wiener process seeds across ε values for variance reduction.
"""
function find_eps_star_ct(p, rng;
                          N_aux  = 200,
                          ε_ref  = 0.1,
                          ε_grid = exp.(range(log(1e-5), log(2.0), length=60)))
    g0 = g_ct(0.0, p.x0, p)

    # Pre-draw N_aux sets of Brownian increments — reused for all ε
    dWs_all = [draw_increments(p, rng) for _ in 1:N_aux]

    m_vals = Float64[]
    for ε in ε_grid
        Gvals = Float64[]
        for i in 1:N_aux
            xT, lp, _ = simulate_guided(ε, p; dWs=dWs_all[i])
            gT = g_ct(p.T, xT, p)
            hT = h_ct(xT, p)
            if !isfinite(lp) || gT + ε <= 0.0 || hT <= 0.0
                push!(Gvals, 0.0)
            else
                G = (hT * (g0 + ε) / (gT + ε)) * exp(-lp)
                push!(Gvals, G^2)
            end
        end
        push!(m_vals, mean(Gvals))
    end
    return ε_grid[argmin(m_vals)], m_vals
end

# ── Run SMC-like IS and compute diagnostics ───────────────────────────────────

"""
Run IS with N paths for a given ε. Returns:
  - log marginal likelihood estimate
  - normalised weights
  - ESS
  - max Qn
"""
function run_IS(ε, N, p, rng)
    log_ws = Float64[]
    for _ in 1:N
        xT, log_psi, _ = simulate_guided(ε, p; rng=rng)
        lw = log_weight(xT, log_psi, ε, p)
        push!(log_ws, isfinite(lw) ? lw : -Inf)
    end
    # Numerically stable normalisation
    lw_max = maximum(log_ws)
    if !isfinite(lw_max)
        # All weights degenerate — return worst-case diagnostics
        return -Inf, fill(1.0/N, N), 1.0, 1.0
    end
    ws     = exp.(log_ws .- lw_max)
    norm_w = ws ./ sum(ws)
    ess    = 1.0 / sum(norm_w.^2)
    max_qn = maximum(norm_w)
    log_ml = lw_max + log(mean(ws))
    return log_ml, norm_w, ess, max_qn
end

# ── Main experiment ───────────────────────────────────────────────────────────

function main(; N=500, R=100, seed=42)
    p = ParaCT()

    println("Parameters: κ=$(p.κ), α=$(p.α), q=$(p.q), T=$(p.T), "*
            "v=$(p.v), σ_obs=$(p.σ_obs)")

    # 1. Plot m̄(ε) for both α=0 (linear) and α=p.α (nonlinear)
    println("\nEstimating m(ε) over grid...")
    εs = exp.(range(log(1e-12), log(1e-5), length=60))

    # Linear case (Psi=1 exactly)
    p_lin = ParaCT(α=0.0)
    rng1  = MersenneTwister(seed)
    ε_star_lin, m_vals_lin = find_eps_star_ct(p_lin, rng1; N_aux=500, ε_grid=εs)
    println("Linear (α=0):    ε* = $ε_star_lin")

    # Nonlinear case
    rng2  = MersenneTwister(seed)
    ε_star, m_vals = find_eps_star_ct(p, rng2; N_aux=500, ε_grid=εs)
    println("Nonlinear (α=$(p.α)): ε* = $ε_star")

    plt_m = plot(log10.(εs), log.(m_vals_lin),
                 xlabel="log10(epsilon)", ylabel="log m(epsilon)",
                 title="mbar(epsilon), continuous-time",
                 label="alpha=0 (linear)", lw=2, color=:blue)
    plot!(plt_m, log10.(εs), log.(m_vals),
          label="alpha=$(p.α) (nonlinear)", lw=2, color=:red)
    vline!(plt_m, [log10(ε_star_lin)], linestyle=:dash, color=:blue)
    vline!(plt_m, [log10(ε_star)],     linestyle=:dash, color=:red)
    savefig(plt_m, "figs/ct_mbar.png")

    # 2. Compare ESS and max-Qn across R runs for several ε values
    ε_range = [0.0, ε_star/10, ε_star, ε_star*10, 0.5]
    ε_labels = ["eps=0", "eps=eps*/10", "eps=eps*", "eps=eps*x10", "eps=0.5"]
    colors   = [:pink, :orange, :blue, :green, :teal]

    ess_runs = Dict(ε => Float64[] for ε in ε_range)
    qn_runs  = Dict(ε => Float64[] for ε in ε_range)
    ll_runs  = Dict(ε => Float64[] for ε in ε_range)

    println("\nRunning $R IS experiments for each ε...")
    for r in 1:R
        for ε in ε_range
            rng_r = MersenneTwister(seed + r)
            log_ml, _, ess, max_qn = run_IS(ε, N, p, rng_r)
            push!(ess_runs[ε],  ess)
            push!(qn_runs[ε],   max_qn)
            push!(ll_runs[ε],   log_ml)
        end
    end

    # ESS CDF
    plt_ess = plot(xlabel="ESS", ylabel="P(ESS <= x)",
                   title="ESS distribution, continuous-time (N=$N, R=$R)",
                   size=(650, 400))
    for (k, ε) in enumerate(ε_range)
        vals = sort(ess_runs[ε])
        cdf  = (1:R) ./ R
        plot!(plt_ess, vals, cdf, label=ε_labels[k], lw=2, color=colors[k])
    end
    savefig(plt_ess, "figs/ct_ess_cdf.png")

    # max-Qn CDF
    plt_qn = plot(xlabel="max Qn", ylabel="P(max Qn <= x)",
                  title="max-Qn distribution, continuous-time (N=$N, R=$R)",
                  size=(650, 400))
    for (k, ε) in enumerate(ε_range)
        vals = sort(qn_runs[ε])
        cdf  = (1:R) ./ R
        plot!(plt_qn, vals, cdf, label=ε_labels[k], lw=2, color=colors[k])
    end
    savefig(plt_qn, "figs/ct_qn_cdf.png")

    # Log-likelihood distribution
    plt_ll = plot(xlabel="log marginal likelihood", ylabel="P(log-lik <= x)",
                  title="Log-likelihood distribution, continuous-time",
                  size=(650, 400))
    for (k, ε) in enumerate(ε_range)
        vals = sort(ll_runs[ε])
        cdf  = (1:R) ./ R
        plot!(plt_ll, vals, cdf, label=ε_labels[k], lw=2, color=colors[k])
    end
    savefig(plt_ll, "figs/ct_ll_cdf.png")

    println("\nSummary:")
    for (k, ε) in enumerate(ε_range)
        ess_ok = filter(isfinite, ess_runs[ε])
        qn_ok  = filter(isfinite, qn_runs[ε])
        n_nan  = R - length(ess_ok)
        println("  $(ε_labels[k]): mean ESS=$(round(mean(ess_ok),digits=1)), "*
                "5th pct ESS=$(round(quantile(ess_ok,0.05),digits=1)), "*
                "mean Qn=$(round(mean(qn_ok),digits=4)), "*
                "NaN/degenerate runs=$n_nan/$R")
    end

    println("\nSaved: figs/ct_mbar.png, figs/ct_ess_cdf.png, "*
            "figs/ct_qn_cdf.png, figs/ct_ll_cdf.png")
end

main()


#include("ct_guided_8_.jl")

function diagnose_weights(; N=20, seed=42)
    p = ParaCT()
    rng = MersenneTwister(seed)

    # find eps*
    eps_star, _ = find_eps_star_ct(p, rng; N_aux=200)
    println("eps* = $eps_star")
    println()

    for ε in [0.0, eps_star]
        println("="^60)
        println("ε = $ε")
        println("="^60)
        rng2 = MersenneTwister(seed)
        for i in 1:N
            xT, log_psi, _ = simulate_guided(ε, p; rng=rng2)
            gT  = g_ct(p.T, xT, p)
            g0  = g_ct(0.0, p.x0, p)
            hT  = h_ct(xT, p)
            lw  = log_weight(xT, log_psi, ε, p)

            @printf("  path %2d: xT=%6.3f  log h=% .2f  log(g0+ε)-log(gT+ε)=% .2f  log_psi=% .2f  lw=% .2f\n",
                i, xT,
                log(max(hT, 1e-300)),
                log(g0 + ε) - log(gT + ε),
                log_psi,
                lw)
        end
        println()
    end
end

using Printf
diagnose_weights()




function estimate_m_grid(ε_grid, N, p, rng)
    @unpack T, n_steps, x0 = p
    dt = T / n_steps
    ts = range(0.0, T, length=n_steps+1)

    m_vals = zeros(length(ε_grid))

    for _ in 1:N
        # simulate one path from P (unguided)
        xs, _ = simulate_unguided(p, rng)
        xT    = xs[end]
        hT    = h_ct(xT, p)
        gT    = g_ct(T, xT, p)
        g0    = g_ct(0.0, x0, p)

        # precompute (b - b_ref) * score_g along the path (ε-independent part)
        db_times_score = [(b(xs[i], p) - b_ref(xs[i], p)) * score_g(ts[i], xs[i], p)
                          for i in 1:n_steps]

        for (k, ε) in enumerate(ε_grid)
            # compute log Ψ(x, ε) by replaying the path
            log_psi = 0.0
            for i in 1:n_steps
                gval     = g_ct(ts[i], xs[i], p)
                r_factor = gval / (gval + ε)
                log_psi += r_factor * db_times_score[i] * dt
            end
            if !isfinite(log_psi)
                continue
            end
            m_vals[k] += ((g0 + ε) * hT^2 / (gT + ε)) * exp(-log_psi)
        end
    end

    return m_vals ./ N
end

pp = ParaCT(α=0.0)
  ε_grid = exp.(range(log(1e-13), log(1e-10), length=60))
  out = estimate_m_grid(ε_grid, 1500, pp, MersenneTwister(10))
plot(ε_grid, out)