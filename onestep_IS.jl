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

using QuadGK          # adaptive 1-D quadrature (fallback / reference)
using FastGaussQuadrature  # Gauss-Hermite nodes & weights
using Distributions   # Normal, Chisq
using Plots

# ── Parameters ────────────────────────────────────────────────────────────────
ω  = 0.0        # intercept of Markov kernel (set as desired)
ψ  = 0.9        # AR coefficient
η  = 1.0        # noise std dev of Markov kernel
#η = 2.5#π/√2 + 0.1
μ  = -1.27      # shift in g
σ² = π^2 / 2   # variance in g
v  = 0.0        # evaluation point (set as desired)

# ── Densities ─────────────────────────────────────────────────────────────────

"""g(x₁) = φ(v; μ + x₁, σ²)  [Gaussian density evaluated at v]"""
function g(x₁)
    return pdf(Normal(μ + x₁, sqrt(σ²)), v)
end

"""
h(x₁) = f_{χ²(1)}(e^{v-x₁}) · e^{v-x₁}

Derived from P(X₁ + log Z² ≤ v) by change of variables.
For large x₁, e^{v-x₁} → 0 and the χ²(1) density f(u) ~ 1/√u / (√(2π)),
so h(x₁) ~ 1/√(e^{v-x₁}).  We handle the limit carefully.
"""
function h(x₁)
    u = exp(v - x₁)           # u = e^{v - x₁} > 0
    # χ²(1) pdf:  f(u) = u^{-1/2} e^{-u/2} / (√(2π))
    fχ = pdf(Chisq(1), u)
    return fχ * u              # = f_{χ²(1)}(u) · e^{v-x₁}
end

# ── Gauss-Hermite integration over P(x₀, dx₁) ────────────────────────────────
# x₁ = μ_kernel + η·t,  P(x₀, dx₁) = φ(t; 0,1) dt
# Gauss-Hermite: ∫ f(t) e^{-t²} dt ≈ Σ wᵢ f(tᵢ)
# Standard normal: ∫ f(t) φ(t) dt = ∫ f(t) e^{-t²/2}/√(2π) dt
#                = (1/√π) ∫ f(√2 s) e^{-s²} ds   (t = √2 s)

const N_GH = 200   # number of Gauss-Hermite nodes

const gh_nodes, gh_weights = gausshermite(N_GH)   # nodes tᵢ, weights wᵢ for e^{-t²}

"""
Integrate f(x₁) against P(x₀, · ) using Gauss-Hermite quadrature.
Returns ∫ f(x₁) P(x₀, dx₁).
"""
function integrate_P(f, x₀)
    μ_kernel = ω + ψ * x₀
    # change of variables: x₁ = μ_kernel + η·√2·s,  t = √2 s
    total = 0.0
    for i in 1:N_GH
        x₁ = μ_kernel + η * sqrt(2) * gh_nodes[i]
        total += gh_weights[i] * f(x₁)
    end
    return total / sqrt(π)   # 1/√π from the Gauss-Hermite normalisation
end

# ── Core quantities ────────────────────────────────────────────────────────────

"""c(x₀) = ∫ g(x₁) P(x₀, dx₁)  — normalising constant of Pᵍ"""
c(x₀) = integrate_P(g, x₀)

"""
I(ε, x₀) = ∫ h(x₁)² / (g(x₁) + ε) · P(x₀, dx₁)
"""
function I_integral(ε, x₀)
    return integrate_P(x₁ -> h(x₁)^2 / (g(x₁) + ε), x₀)
end

"""
m(ε, x₀) = (c(x₀) + ε) · I(ε, x₀)
"""
function m(ε, x₀)
    return (c(x₀) + ε) * I_integral(ε, x₀)
end

"""
m̄(ε, x₀) = log m(ε, x₀)
"""
m_bar(ε, x₀) = log(m(ε, x₀))

# ── Derivative via finite differences (optional) ──────────────────────────────

"""d/dε  m̄(ε, x₀)  by central finite differences."""
function dm_bar_dε(ε, x₀; δ=1e-5)
    return (m_bar(ε + δ, x₀) - m_bar(ε - δ, x₀)) / (2δ)
end

# ── Demo / plotting ───────────────────────────────────────────────────────────

function main(η)
    x₀ = 0.0   # fixed conditioning value; change as needed

    println("Parameters:")
    println("  ω=$ω, ψ=$ψ, η=$η, μ=$μ, σ²=$σ², v=$v, x₀=$x₀")
    println()

    cx₀ = c(x₀)
    println("c(x₀) = $cx₀")

    εs = range(0.0, 0.1, length=200)

    m_vals    = [m(ε, x₀)     for ε in εs]
    mbar_vals = [m_bar(ε, x₀) for ε in εs]

    # println("\nSample values:")
    # println("  ε=0.0 : m = $(m(0.0, x₀)),  m̄ = $(m_bar(0.0, x₀))")
    # println("  ε=0.5 : m = $(m(0.5, x₀)),  m̄ = $(m_bar(0.5, x₀))")
    # println("  ε=1.0 : m = $(m(1.0, x₀)),  m̄ = $(m_bar(1.0, x₀))")

    p1 = plot(εs, mbar_vals,
              xlabel="ε", ylabel="m̄(ε)",
              title="m̄(ε) = log m(ε)  [x₀ = $x₀], η=$η",
              lw=2, legend=false)

    p2 = plot(εs, dm_bar_dε.(εs, x₀),
              xlabel="ε", ylabel="d/dε  m̄(ε)",
              title="Derivative of m̄(ε)",
              lw=2, legend=false, color=:orange)
    savefig(p1, "mbar_plot_$η.png")
    #plt = plot(p1, p2, layout=(1,2), size=(900, 400))
    
    #savefig(p2, "dmbar_plot.png")
    #savefig(plt, "mbar_plot.png")
    println("\nPlot saved to mbar_plot.png")

    return εs, m_vals, mbar_vals
end

main(η)


