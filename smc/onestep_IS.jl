wd = @__DIR__
cd(wd)

"""
Numerical evaluation of ε ↦ m̄(ε) = log m(ε), where

    m(ε) = (c(x₀) + ε) · ∫ h(x₁)² / (g(x₁) + ε) · P(x₀, dx₁)

with
    P(x₀, dx₁)  = N(x₁; ω + ψ x₀, η²)          Markov kernel
    g(x₁)       = φ(v; μ + x₁, σ²)              = φ(x₁; v - μ, σ²)
    h(x₁)       = f_{χ²(1)}(e^{v-x₁}) · e^{v-x₁}
    c(x₀)       = ∫ P(x₀, dx₁) g(x₁) dx₁       (normalising constant for Pᵍ)

All integrals are over x₁ ∈ ℝ and are evaluated by Gauss-Hermite quadrature
(change of variables x₁ = ω + ψ x₀ + η·t, so P(x₀, dx₁) = N(t;0,1) dt·η).
"""

using FastGaussQuadrature  # Gauss-Hermite nodes & weights
using Distributions        # Normal, Chisq
using Plots
using Parameters
using UnPack

# ── Parameters ────────────────────────────────────────────────────────────────
@with_kw struct Para{R}
    x₀::R = 0.0
    ω::R  = 0.0        # intercept of Markov kernel
    ψ::R  = 0.9        # AR coefficient
    η::R  = 3.0        # noise std dev of Markov kernel
    μ::R  = -1.27      # shift in g
    σ²::R = π^2 / 2   # variance in g
    v::R  = 0.0        # observation
end

# ── Densities ─────────────────────────────────────────────────────────────────

"""g(x₁, p) = φ(v; μ + x₁, σ²)  [Gaussian density evaluated at v]"""
function g(x₁, p)
    @unpack μ, σ², v = p
    return pdf(Normal(μ + x₁, sqrt(σ²)), v)
end

"""
h(x₁, p) = f_{χ²(1)}(e^{v-x₁}) · e^{v-x₁}

Derived from P(X₁ + log Z² ≤ v) by change of variables.
For large x₁, e^{v-x₁} → 0 and h(x₁) ~ √(e^{v-x₁}).
"""
function h(x₁, p)
    @unpack v = p
    u  = exp(v - x₁)
    fχ = pdf(Chisq(1), u)
    return fχ * u
end

# ── Gauss-Hermite integration over P(x₀, dx₁) ────────────────────────────────
# x₁ = μ_kernel + η·√2·s,  P(x₀, dx₁) = φ(t; 0,1) dt
# GH: ∫ f(t) e^{-t²} dt ≈ Σ wᵢ f(tᵢ)
# Standard normal: ∫ f(t) φ(t) dt = (1/√π) ∫ f(√2 s) e^{-s²} ds

const N_GH = 100
const gh_nodes, gh_weights = gausshermite(N_GH)

"""
Integrate f(x₁) against P(x₀, ·) using Gauss-Hermite quadrature.
Returns ∫ f(x₁) P(x₀, dx₁).
"""
function integrate_P(f, p)
    @unpack ω, ψ, x₀, η = p
    μ_kernel = ω + ψ * x₀
    total = 0.0
    for i in 1:N_GH
        x₁     = μ_kernel + η * sqrt(2) * gh_nodes[i]
        total += gh_weights[i] * f(x₁)
    end
    return total / sqrt(π)
end

# ── Core quantities ────────────────────────────────────────────────────────────

"""c(p) = ∫ g(x₁, p) P(x₀, dx₁)  — normalising constant of Pᵍ"""
c(p) = integrate_P(x₁ -> g(x₁, p), p)

"""I(ε, p) = ∫ h(x₁)² / (g(x₁) + ε) · P(x₀, dx₁)"""
I_integral(ε, p) = integrate_P(x₁ -> h(x₁, p)^2 / (g(x₁, p) + ε), p)

"""m(ε, p) = (c(p) + ε) · I(ε, p)"""
m(ε, p) = (c(p) + ε) * I_integral(ε, p)

"""m̄(ε, p) = log m(ε, p)"""
m_bar(ε, p) = log(m(ε, p))

"""d/dε m̄(ε, p) by central finite differences"""
function dm_bar_dε(ε, p; δ=1e-5)
    return (m_bar(ε + δ, p) - m_bar(ε - δ, p)) / (2δ)
end

# ── Demo / plotting ───────────────────────────────────────────────────────────

function main(εs, p)
    @unpack ω, ψ, η, μ, σ², v, x₀ = p
    println("Parameters:")
    println("  ω=$ω, ψ=$ψ, η=$η, μ=$μ, σ²=$σ², v=$v, x₀=$x₀")
    println()

    println("c(x₀) = $(c(p))")

    m_vals    = [m(ε, p)     for ε in εs]
    mbar_vals = [m_bar(ε, p) for ε in εs]

    p1 = plot(collect(εs), mbar_vals,
              xlabel="ε", ylabel="mbar",
              title="m̄(ε), η=$(η)",
              lw=2, legend=false)

    savefig(p1, "figs/mbar_plot_$(η).png")
    println("Plot saved to mbar_plot_$(η).png")

    return m_vals, mbar_vals
end

# ── Run ───────────────────────────────────────────────────────────────────────

p = Para()

for η_val in [1.0, 3.0]
    if η_val<1.1
        εs = range(0.00, .05, length=200)
    else
        εs = exp.(range(log(1e-12), log(.001), length=200))
    end
    p = Para(η=η_val)
    main(εs, p)
end