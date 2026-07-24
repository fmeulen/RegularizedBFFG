wd = @__DIR__
cd(wd)

using Plots
"""
Continuous-time guided proposals with ε-regularisation.

SDE:    dX_t = b(t, X_t) dt + q dW_t,   X_0 = x0

Observation (log-chi-squared, same as discrete-time case):
  v = x_T + log(Z²/2),  Z ~ N(0,1)
  h(x_T) = f_{χ²(1)}(e^{v-x_T}) · e^{v-x_T}
          = (2π)^{-1/2} exp(-(v-x_T)/2) exp(-e^{v-x_T}/2)

Gaussian approximation to h (same as discrete-time):
  g(T, x) = φ(x; v - μ_χ, σ_χ²)
  μ_χ = -1.27,  σ_χ² = π²/2

Backward ODE for g(t,x) = φ(x; μ(t), Σ(t)) under reference dX = -κX dt + q dW:
  μ(T) = v - μ_χ,  Σ(T) = σ_χ²
  μ(t) = μ(T) · exp(-κ(T-t))
  Σ(t) = (Σ(T) + q²/2κ) exp(2κ(T-t)) - q²/2κ   [always > 0 since Σ(T)=π²/2≈4.93]

Condition Σ(t)>0 holds as long as q² < 2κ·π²/2 = κπ², i.e. q < π√κ ≈ 3.14 for κ=1.

m(ε) = (g(x₀)+ε) · E_P[ h(x_T)² / (g(x_T)+ε) · Ψ(x,ε)⁻¹ ]
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

"""
h(x_T) = f_{χ²(1)}(e^{v-x_T}) · e^{v-x_T}
        = (2π)^{-1/2} exp(-(v-x_T)/2) exp(-e^{v-x_T}/2)
"""
function h_ct(xT, p)
    u = p.v - xT          # u = v - x_T
    return (2π)^(-0.5) * exp(-u/2) * exp(-exp(u)/2)
end

function h(x₁, p) # original
    @unpack v = p
    u  = exp(v - x₁)
    fχ = pdf(Chisq(1), u)
    return fχ * u
end



# ── Backward ODE for g(t,x) = φ(x; μ(t), Σ(t)) ──────────────────────────────
# Terminal condition: g(T,x) = φ(x; v-μ_χ, σ_χ²)
# μ(T) = v - μ_χ,  Σ(T) = σ_χ²
# μ̇ = -κμ  →  μ(t) = μ(T) exp(κ(T-t))
# Σ̇ = -2κΣ + q²  →  Σ(t) = (Σ(T) + q²/2κ) exp(2κ(T-t)) - q²/2κ

function gaussian_g_params(t, p)
    @unpack κ, q, v, T, μ_χ, σ_χ² = p
    μT = v - μ_χ
    ΣT = σ_χ²
    μt = μT * exp(κ * (T - t))
    Σt = -q^2/(2κ) + (ΣT + q^2/(2κ)) * exp(2κ*(T-t))
    return μt, Σt
end

function g_ct(t, x, p)
    μt, Σt = gaussian_g_params(t, p)
    return pdf(Normal(μt, sqrt(Σt)), x)
end

function score_g(t, x, p)
    μt, Σt = gaussian_g_params(t, p)
    return -(x - μt) / Σt
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
        hT = h_ct(xT, p)
        gT = g_ct(p.T, xT, p)
        h2 = hT^2

        if h2 == 0.0 || !isfinite(h2)
            continue
        end

        for (k, ε) in enumerate(ε_grid)
            denom = gT + ε
            if denom <= 0.0
                continue
            end
            log_psi = log_psi_from_path(xs, ε, p)
            if !isfinite(log_psi)
                continue
            end
            contrib = h2 / denom * exp(-log_psi)
            if isfinite(contrib)
                sums[k]   += contrib
                counts[k] += 1
            end
        end
    end
    return [(g0 + ε_grid[k]) * sums[k] / max(counts[k], 1) for k in 1:K]
end


# ── Main ──────────────────────────────────────────────────────────────────────

function main(ψ, η, T, ε_grid; N=1000, seed=422)
    κ0 = -log(ψ) / T
    q0 = sqrt(2κ0 * η^2/(1.0 -ψ^2))

    p_lin = ParaCT(α=0.0, κ=κ0, q=q0)
    p_nln = ParaCT(α=0.5, κ=κ0, q=q0)

    println("Parameters: κ=$(p_lin.κ), q=$(p_lin.q), T=$(p_lin.T)")

    println("Estimating m(ε) with N=$N unguided paths...")
    rng1 = MersenneTwister(seed)
    m_lin = estimate_m_grid(ε_grid, N, p_lin, rng1)

    rng2 = MersenneTwister(seed)
    m_nln = estimate_m_grid(ε_grid, N, p_nln, rng2)

    # exclude ε=0 from argmin search (index 1) for display
    ε_star_lin = ε_grid[1 + argmin(m_lin[2:end])]
    ε_star_nln = ε_grid[1 + argmin(m_nln[2:end])]
    println("Linear (α=0):      ε* = $ε_star_lin")
    println("Nonlinear (α=0.5): ε* = $ε_star_nln")
    println()
    println("m(0)   linear:    $(m_lin[1])")
    println("m(ε*)  linear:    $(m_lin[1 + argmin(m_lin[2:end])])")
    println("m(0)   nonlinear: $(m_nln[1])")
    println("m(ε*)  nonlinear: $(m_nln[1 + argmin(m_nln[2:end])])")

    # plot excluding ε=0 point (which may be Inf)
    idx = 1:length(ε_grid)
    plt = plot(ε_grid[idx], log.(m_lin[idx]),
               xlabel="ε", ylabel="log m(ε)",
               title="m̄(ε), continuous-time, log-χ² observation (N=$N)",
               label="α=0 (linear, Ψ≡1)", lw=2, color=:blue)
               #xscale=:log10)
    plot!(plt, ε_grid[idx], log.(m_nln[idx]),
          label="α=0.5 (nonlinear)", lw=2, color=:red)
    vline!(plt, [ε_star_lin], linestyle=:dash, color=:blue, label="ε* linear")
    vline!(plt, [ε_star_nln], linestyle=:dash, color=:red,  label="ε* nonlinear")
    savefig(plt, "figs/ct_mbar_v3.png")
    println("\nSaved figs/ct_mbar_v3.png")
end




T = 1.0
ψ = 0.9 # as before, discrete-time case 



# rng = MersenneTwister(11)
# ε_grid = vcat(0.0, exp.(range(log(1e-5), log(2.0), length=10)))
 p_lin = ParaCT(α=0.0)
# p_nln = ParaCT(α=0.5)
 p = p_lin
# estimate_m_grid(ε_grid, 50, p, rng);


# two grids: coarse for overview, fine near origin to see descent
η = 3.0 
ε_grid = vcat(0.0, exp.(range(log(1e-8), log(1e-1), length=80)))

η = 1.0
ε_grid = range(0.0, 0.05, length=20)

main(ψ, η, T, ε_grid)

