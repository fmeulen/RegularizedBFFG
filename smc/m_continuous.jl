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

Backward ODE for g(t,x) = A(t) φ(x; μ(t), Σ(t)) under reference dX = -κX dt + q dW:
  μ(T) = v - μ_χ,  Σ(T) = σ_χ²
  
m(ε) = (g(x₀)+ε) · E_P[ h(x_T)² / (g(x_T)+ε) · Ψ(x,ε) ]
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

@with_kw struct ParaCTT{R}
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
Estimate m(ε) = (g(x₀)+ε) · E_P[ h(x_T)² / (g(x_T)+ε) · Ψ(x,ε)⁻ ]
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
            contrib = h2 / denom * exp(log_psi) # sign wrong????
            sums[k]   += contrib
            counts[k] += 1
        end
    end
    return [(g0 + ε_grid[k]) * sums[k] / max(counts[k], 1) for k in 1:K]
end


# ── Main ──────────────────────────────────────────────────────────────────────

# constructor 
function ParaCTT(ψ, η, α, T) 
    κ0 = -log(ψ) / T
    q0 = sqrt(2κ0 * η^2/(1.0 - ψ^2))
    ParaCTT(κ=κ0, q=q0, α=α, T=T)
end


function main(ψ, η, α, T, ε_grid; N=1000, seed=422)
    p = ParaCTT(ψ, η, α, T)
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
εs = exp.(range(log(1e-12), log(1e-8), length=200))
#εs = range(0.00, 4.0, length=200)
m3, plt3  = main(ψ, η,α, T, εs; N=200)
plt3
plt3b = plot(εs[:end], log.(m3)[10:end],    xlabel="ε", ylabel="log m(ε)",
               title="m̄(ε), η=$η, α=$α",
               label="", lw=2, color=:blue)


plot(plt1, plt2, plt3b, layout= @layout [a;b;c])
  savefig("figs/mbar_plots_ct.png")



 CHECKING = false
 if CHECKING 
    ## checks discrete vs continuous
    p_discr = Para()
    p

    # match for continuous time case
    κ0 = -log(p_discr.ψ) / T
    q0 = sqrt(2κ0 * p_discr.η^2/(1.0 -p_discr.ψ^2))

    p = ParaCTT(κ=κ0, α=0.0, q=q0)

    g_ct(0.0, p.x0, p)
    c(p_discr)

    xT = 0.98
    g_ct(p.T, xT, p)
    g(xT, p_discr)  
 end











