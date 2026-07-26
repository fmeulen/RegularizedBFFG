wd = @__DIR__
cd(wd)

using Plots
"""
Continuous-time guided proposals with ε-regularisation.

SDE:    dX_t = b(t, X_t) dt + q dW_t,   X_0 = x0

Observation (log-chi-squared, same as discrete-time case):
  v = x_T + log(Z²/2),  Z ~ N(0,1)
  h(x_T) = f_{χ²(1)}(e^{v-x_T}) · e^{v-x_T}
          = (2π)^{-1/2} exp((v-x_T)/2) exp(-e^{v-x_T}/2)

Gaussian approximation to h (same as discrete-time):
  g(T, x) = φ(x; v - μ_χ, σ_χ²)
  μ_χ = -1.27,  σ_χ² = π²/2

Backward ODE for g(t,x) = φ(x; μ(t), Σ(t)) under reference dX = -κX dt + q dW:
  μ(T) = v - μ_χ,  Σ(T) = σ_χ²
  μ(t) = μ(T) · exp(κ(T-t))
  Σ(t) = (Σ(T) + q²/2κ) exp(2κ(T-t)) - q²/2κ   [always > 0 since Σ(T)=π²/2≈4.93]

m(ε) = (g(x₀)+ε) · E_P[ h(x_T)² / (g(x_T)+ε) · Ψ(x,ε)⁻¹ ]
"""

using Distributions
using Random
using Statistics
using Plots
using Parameters
using UnPack
using LogExpFunctions
using QuadGK

mkpath("figs")




# ── Parameters ────────────────────────────────────────────────────────────────

@with_kw struct ParaCT{R}
    x0::R    = 0.0
    κ::R     = 1.0
    α::R     = 0.5        # nonlinear amplitude; α=0 → linear (Ψ≡1)
    q::R     = 4.0
    v::R     = 0.0        # observed value
    T::R     = 1.0
    n_steps::Int = 1000
    # log-chi-squared approximation parameters (fixed, used in the backward filtering where h_T is approximated by a N(μ_χ, σ_χ²))
    μ_χ::R   = -1.27
    σ_χ²::R  = π^2 / 2
end

# ── Drifts ────────────────────────────────────────────────────────────────────

b(x, p)     = -p.κ * x + p.α * sin(x)
b_ref(x, p) = -p.κ * x

# ── Observation density h (log-chi-squared) ───────────────────────────────────

function h_ct(x₁, p) # original
    @unpack v = p
    u  = exp(v - x₁)
    fχ = pdf(Chisq(1), u)
    return fχ * u
end

function g_ct(t, x, p)
    @unpack κ, q, v, T, μ_χ, σ_χ² = p
    τ = T - t
    η²_τ = q^2/(2κ) * (1 - exp(-2κ*τ))
    return pdf(Normal(v - μ_χ, sqrt(σ_χ² + η²_τ)), exp(-κ*τ) * x)
end

function score_g(t, x, p)
    @unpack κ, q, v, T, μ_χ, σ_χ² = p
    τ = T - t
    η²_τ = q^2/(2κ) * (1 - exp(-2κ*τ))
    return -exp(-κ*τ) * (exp(-κ*τ)*x - (v-μ_χ)) / (σ_χ² + η²_τ)
end

function r_eps(t, x, ε, p)
    gval = g_ct(t, x, p)
    return gval / (gval + ε) * score_g(t, x, p)
end


# ── Unguided simulation ───────────────────────────────────────────────────────

function simulate_unguided(p, rng)
    @unpack x0, q, T, n_steps = p
    dt   = T / n_steps
    sqdt = sqrt(dt)
    xs   = Vector{Float64}(undef, n_steps + 1)
    xs[1] = x0
    for i in 1:n_steps
        dW      = randn(rng) * sqdt
        xs[i+1] = xs[i] + b(xs[i], p) * dt + q * dW
    end
    return xs
end

# ── log Ψ along a stored path ─────────────────────────────────────────────────

function log_psi_from_path(xs, ε, p)
    @unpack T, n_steps = p
    dt      = T / n_steps
    ts      = range(0.0, T, length=n_steps+1)
    log_psi = 0.0
    for i in 1:n_steps
        x    = xs[i]
        db   = b(x, p) - b_ref(x, p)    # = α sin(x)
        rval = r_eps(ts[i], x, ε, p)
        log_psi += db * rval * dt
        if !isfinite(log_psi)
            return -Inf
        end
    end
    return log_psi
end

# ── m(ε) estimator integrating against P ─────────────────────────────────────

"""
Estimate m(ε) = (g(x₀)+ε) · E_P[ h(x_T)² / (g(x_T)+ε) · Ψ(x,ε)⁻¹ ]
Simulate N paths from P once, replay each for all ε in ε_grid.
"""
function estimate_m_grid(ε_grid, N, p, rng)
    g0     = g_ct(0.0, p.x0, p)
    K      = length(ε_grid)
    sums   = zeros(K)
    counts = zeros(Int, K)

   for _ in 1:N
        xs = simulate_unguided(p, rng)
        xT = xs[end]

        # skip exploded paths
        if !isfinite(xT) || any(!isfinite, xs)
            continue
        end

        hT = h_ct(xT, p)
        gT = g_ct(p.T, xT, p)
        h2 = hT^2

     
        for (k, ε) in enumerate(ε_grid)
            denom = gT + ε
            log_psi = log_psi_from_path(xs, ε, p)
            contrib = h2 / denom * exp(-log_psi)
            sums[k]   += contrib
            counts[k] += 1
        end
    end
    return [(g0 + ε_grid[k]) * sums[k] / max(counts[k], 1) for k in 1:K]
end


# ── Main ──────────────────────────────────────────────────────────────────────

# constructor 
function ParaCT(ψ, η, α, T) 
    κ0 = -log(ψ) / T
    q0 = sqrt(2κ0 * η^2/(1.0 - ψ^2))
    ParaCT(κ=κ0, q=q0, α=α, T=T)
end


ParaCT(0.9, 1.0, 0.0, 1.0)

function main(ψ, η, α, T, ε_grid; N=1000, seed=422)
    p = ParaCT(ψ, η, α, T)
    println("Parameters: κ=$(p.κ), q=$(p.q), T=$(p.T), α=$(p.α)")

    println("Estimating m(ε) with N=$N unguided paths...")
    rng1 = MersenneTwister(seed)
    m = estimate_m_grid(ε_grid, N, p, rng1)

    # exclude ε=0 from argmin search (index 1) for display
    ε_star = ε_grid[1 + argmin(m[2:end])]
    println("ε* = $ε_star")
    println()
 
    # plot excluding ε=0 point (which may be Inf)
    idx = 1:length(ε_grid)
    plt = plot(ε_grid[idx], log.(m[idx]),
               xlabel="ε", ylabel="log m(ε)",
               title="m̄(ε), η=$η, α=$α",
               label="", lw=2, color=:blue)
    vline!(plt, [ε_star], linestyle=:dash, color=:blue, label="")
    m, plt 
end

T = 1.0
ψ = 0.9

# case 1 
α = 0.0
η = 1.0
εs = range(0.00, .05, length=200)
m1, plt1 = main(ψ, η,α, T, εs; N=2000)
plt1

# case 2
α = 0.0
η = 3.0
εs = exp.(range(log(1e-12), log(.001), length=200))
m2, plt2 = main(ψ, η, α, T, εs; N=20000)
plt2

# case 3
α = 1.5
η = 1.0
εs = range(0.00, 2.0, length=200)
m3, plt3  = main(ψ, η,α, T, εs; N=2000)
plt3
plot(εs[50:end], log.(m3)[50:end])


plot(plt1, plt2, plt3, layout= @layout [a;b;c])




# a check
η = 3.0
κ0 = -log(ψ) / T
q0 = sqrt(2κ0 * η^2/(1.0 -ψ^2))
p_lin_3 = ParaCT(α=0.0, κ=κ0, q=q0)
using QuadGK
I0, _ = quadgk(x -> h_ct(x, p_lin_3)^2 / g_ct(p_lin_3.T, x, p_lin_3) * 
               pdf(Normal(p_lin_3.x0 * exp(-p_lin_3.κ*p_lin_3.T), 
               sqrt(p_lin_3.q^2/(2p_lin_3.κ)*(1-exp(-2p_lin_3.κ*p_lin_3.T)))), x), 
               -Inf, Inf)
println("I(0) = $I0")
println("m(0) = $(g_ct(0.0, p_lin_3.x0, p_lin_3) * I0)")




## checks discrete vs continuous
p_discr = Para()
p

# match for continuous time case
κ0 = -log(p_discr.ψ) / T
q0 = sqrt(2κ0 * p_discr.η^2/(1.0 -p_discr.ψ^2))

p = ParaCT(κ=κ0, α=0.0, q=q0)

g_ct(0.0, p.x0, p)
c(p_discr)

xT = 0.98
g_ct(p.T, xT, p)
g(xT, p_discr)  















## below is by drawing from P^{g+\epsilon}

function draw_increments(p, rng)
    @unpack T, n_steps, q = p
    dt   = T / n_steps
    sqdt = sqrt(dt)
    return randn(rng, n_steps) .* sqdt
end

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
        db         = b(x, p) - b_ref(x, p)    # = α sin(x)
        log_psi   += db * rval * dt

        if !isfinite(log_psi) || abs(log_psi) > 1e6 || !isfinite(x) || abs(x) > 1e4
            log_psi = -1e6
            xs[i+1] = x
            for j in i+2:n_steps+1; xs[j] = xs[j-1]; end
            break
        end

        xs[i+1] = x + drift_guid * dt + q * dW
    end

    return xs[end], log_psi, xs
end


function log_weight(xT, log_psi, ε, p)
    g0 = g_ct(0.0, p.x0, p)
    gT = g_ct(p.T, xT, p)
    hT = h_ct(xT, p)
    if hT <= 0.0 || (gT + ε) <= 0.0 || (g0 + ε) <= 0.0
        return -Inf
    end
    return log(hT) + log(g0 + ε) - log(gT + ε) - log_psi
end

function estimate_m_grid(ε_grid, N, p, rng)
    K      = length(ε_grid)
    sums   = zeros(K)
    counts = zeros(Int, K)

    # draw N Brownian paths once
    dWs_all = [draw_increments(p, rng) for _ in 1:N]

    for i in 1:N
        for (k, ε) in enumerate(ε_grid)
            xT, log_psi, _ = simulate_guided(ε, p; dWs=dWs_all[i])
            lw = log_weight(xT, log_psi, ε, p)
            G2 = exp(2*lw)
            if isfinite(G2)
                sums[k]   += G2
                counts[k] += 1
            end
        end
    end

    return [sums[k] / max(counts[k], 1) for k in 1:K]
end

# ── Test ──────────────────────────────────────────────────────────────────────

seed = 5
N = 2000

function test_estimate_m_grid(; N=5000, seed=42)
    T = 1.0
    ψ = 0.9
    κ0 = -log(ψ) / T

    println("=== Test 1: α=0, η=1 (finite variance, should show U-shape) ===")
    η = 1.0
    q0 = sqrt(2κ0 * η^2 / (1 - ψ^2))
    p = ParaCT(α=0.0, κ=κ0, q=q0)
    ε_grid = vcat(0.0, exp.(range(log(1e-6), log(0.5), length=60)))
    rng = MersenneTwister(seed)
    m1 = estimate_m_grid(ε_grid, N, p, rng)
    idx_star = argmin(m1)
    println("  m(0)  = $(m1[1])")
    println("  ε*    = $(ε_grid[idx_star])")
    println("  m(ε*) = $(m1[idx_star])")
    println("  U-shape: $(m1[1] > m1[idx_star] && m1[end] > m1[idx_star])")

    # cross-check m(0) against discrete-time I(0) via quadrature
    p_discr = Para(η=η)
    cg_val = c(p_discr)
    I0, _ = quadgk(
        x -> begin
            hval = h_ct(x, p)
            gval = g_ct(p.T, x, p)
            pval = pdf(Normal(p.x0*exp(-p.κ*p.T),
                       sqrt(p.q^2/(2p.κ)*(1-exp(-2p.κ*p.T)))), x)
            (gval <= 0.0 || !isfinite(hval)) ? 0.0 : hval^2/gval * pval
        end, -Inf, Inf; rtol=1e-8)
    println("  m(0) quadrature = $(cg_val * I0)")
    println("  m(0) MC         = $(m1[1])")

    plt1 = plot(ε_grid[2:end], log.(m1[2:end]),
                xlabel="ε", ylabel="log m(ε)",
                title="m̄(ε), α=0, η=1.0 (N=$N)",
                lw=2, color=:blue, xscale=:log10)
    vline!(plt1, [ε_grid[idx_star]], linestyle=:dash, color=:blue, label="ε*")
    savefig(plt1, "figs/ct_mbar_eta1_test.png")

    println()
    println("=== Test 2: α=0, η=3 (infinite variance, m(0)=∞) ===")
    η = 3.0
    q1 = sqrt(2κ0 * η^2 / (1 - ψ^2))
    p3 = ParaCT(α=0.0, κ=κ0, q=q1)
    ε_grid3 = vcat(0.0, exp.(range(log(1e-12), log(0.001), length=60)))
    rng = MersenneTwister(seed)
    m3 = estimate_m_grid(ε_grid3, N, p3, rng)
    idx_star3 = argmin(m3[2:end]) + 1
    println("  m(0)  = $(m3[1])")   # should be very large or Inf
    println("  ε*    = $(ε_grid3[idx_star3])")
    println("  m(ε*) = $(m3[idx_star3])")
    println("  U-shape: $(m3[1] > m3[idx_star3] && m3[end] > m3[idx_star3])")

    # cross-check m(0) against discrete-time I(0) via quadrature
    p_discr = Para(η=η)
    cg_val = c(p_discr)
    I0, _ = quadgk(
        x -> begin
            hval = h_ct(x, p)
            gval = g_ct(p.T, x, p)
            pval = pdf(Normal(p.x0*exp(-p.κ*p.T),
                       sqrt(p.q^2/(2p.κ)*(1-exp(-2p.κ*p.T)))), x)
            (gval <= 0.0 || !isfinite(hval)) ? 0.0 : hval^2/gval * pval
        end, -Inf, Inf; rtol=1e-8)
    println("  m(0) quadrature = $(cg_val * I0)")
    println("  m(0) MC         = $(m1[1])")

    plt3 = plot(ε_grid3[2:end], log.(m3[2:end]),
                xlabel="ε", ylabel="log m(ε)",
                title="m̄(ε), α=0, η=3.0 (N=$N)",
                lw=2, color=:red, xscale=:log10)
    vline!(plt3, [ε_grid3[idx_star3]], linestyle=:dash, color=:red, label="ε*")
    savefig(plt3, "figs/ct_mbar_eta3_test.png")

    println()
    println("=== Test 3: α=0.5, η=3 (nonlinear, Ψ≠1) ===")
    p_nln = ParaCT(α=0.5, κ=κ0, q=q1)
    rng = MersenneTwister(seed)
    m_nln = estimate_m_grid(ε_grid3, N, p_nln, rng)
    idx_star_nln = argmin(m_nln[2:end]) + 1
    println("  m(0)  = $(m_nln[1])")
    println("  ε*    = $(ε_grid3[idx_star_nln])")
    println("  m(ε*) = $(m_nln[idx_star_nln])")

    plt3_nl = plot(ε_grid3[2:end], log.(m_nln[2:end]),
          label="α=0.5", lw=2, color=:orange, xscale=:log10)
    vline!(plt3_nl, [ε_grid3[idx_star_nln]], linestyle=:dash, color=:orange)
    savefig(plt3_nl, "figs/ct_mbar_eta3_nl_test.png")

    println("\nSaved: figs/ct_mbar_eta1_test.png, figs/ct_mbar_eta3_test.png")
end

test_estimate_m_grid()