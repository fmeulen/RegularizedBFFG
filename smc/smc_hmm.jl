wd = @__DIR__
cd(wd)

"""
SMC for the HMM using fully closed-form expressions.

Model:
    X_t | X_{t-1}  ~  N(ω + ψ X_{t-1}, η²)
    V_t | X_t       =  X_t + log Z_t²,  Z_t ~ N(0,1)

Twisted proposal P^{g_t + ε} is a mixture:
    P^{g+ε}(x, ·) = α(x) · P^g(x, ·) + (1-α(x)) · P(x, ·)

where
    α(x)      = c(x) / (c(x) + ε)
    P^g(x, ·) = N(·; μ*(x), (σ*)²)                    [closed form]
    c(x)      = N(vt - μ; ω + ψx, η² + σ²)            [closed form]

IS weight:
    G_ε(x, x₁) = h(x₁) · (c(x) + ε) / (g(x₁) + ε)
"""

include("onestep_IS.jl")   # loads Para, g, h, and all one-step functions

using Random
using Optim
using Statistics

# ── Closed-form expressions ───────────────────────────────────────────────────

"""
c_closed(x, vt, p) = ∫ P(x, dx₁) g_t(x₁)  =  N(vt - μ; ω + ψx, η² + σ²)
"""
function c_closed(x, vt, p)
    @unpack ω, ψ, η, μ, σ² = p
    return pdf(Normal(ω + ψ*x, sqrt(η^2 + σ²)), vt - μ)
end

"""
Parameters of P^g(x, ·) = N(·; μ*, σ*):
  1/σ*² = 1/η² + 1/σ²
  μ*    = σ*² · ((ω + ψx)/η² + (vt - μ)/σ²)
"""
function Pg_params(x, vt, p)
    @unpack ω, ψ, η, μ, σ² = p
    σ²_star = 1.0 / (1.0/η^2 + 1.0/σ²)
    μ_star  = σ²_star * ((ω + ψ*x)/η^2 + (vt - μ)/σ²)
    return μ_star, sqrt(σ²_star)
end

"""g_t(x₁, vt, p) = N(x₁; vt - μ, σ²)"""
function g_t(x₁, vt, p)
    @unpack μ, σ² = p
    return pdf(Normal(vt - μ, sqrt(σ²)), x₁)
end

"""h_t(x₁, vt, p) = f_{χ²(1)}(e^{vt-x₁}) · e^{vt-x₁}"""
function h_t(x₁, vt, p)
    u = exp(vt - x₁)
    return pdf(Chisq(1), u) * u
end

"""
Sample from P^{g+ε}(x, ·):
  with prob α = c(x)/(c(x)+ε): draw from P^g  = N(μ*, σ*)
  with prob 1-α:                draw from P    = N(ω+ψx, η)
"""
function sample_twisted(x, vt, ε, p, rng)
    @unpack ω, ψ, η = p
    ct = c_closed(x, vt, p)
    α  = ct / (ct + ε)
    if rand(rng) < α
        μ_star, σ_star = Pg_params(x, vt, p)
        return randn(rng) * σ_star + μ_star
    else
        return randn(rng) * η + ω + ψ*x
    end
end

# ── Simulate HMM ─────────────────────────────────────────────────────────────

function simulate_hmm(x0, T, p; rng=Random.default_rng())
    @unpack ω, ψ, η = p
    xs = Vector{Float64}(undef, T+1)
    vs = Vector{Float64}(undef, T)
    xs[1] = x0
    for t in 1:T
        xs[t+1] = ω + ψ*xs[t] + η*randn(rng)
        Z = randn(rng)
        vs[t] = xs[t+1] + log(Z^2)
    end
    return xs, vs
end

# ── Resampling ────────────────────────────────────────────────────────────────

function systematic_resample(weights, N, rng)
    u    = rand(rng) / N
    idx  = Vector{Int}(undef, N)
    cumw = cumsum(weights)
    j    = 1
    for i in 1:N
        target = u + (i-1)/N
        while j < N && cumw[j] < target
            j += 1
        end
        idx[i] = j
    end
    return idx
end

# ── SMC following Chopin-Papaspiliopoulos Algorithm 10.5 ─────────────────────

"""
Run GPF (Algorithm 10.5 of Chopin-Papaspiliopoulos) with twisted proposal
P^{g+ε}. Tracks ancestor indices and carry-over weights exactly as in the
book, so the likelihood increment is correct whether or not resampling occurs.

Returns:
  log_liks  : incremental log p̂(v_t | v_{1:t-1})  via eq. (10.3)
  ess_vec   : ESS at each step
  min_ess   : minimum ESS over all steps
  particles : final particle cloud X_T^{1:N}
"""
function run_smc(vs, N, ε, p; rng=Random.default_rng(), ess_min=N/2)
    @unpack ψ, η = p
    T      = length(vs)
    σ_stat = η / sqrt(1 - ψ^2)

    X     = σ_stat * randn(rng, N)
    w     = fill(1.0/N, N)
    w_raw = fill(1.0,   N)

    log_liks = Vector{Float64}(undef, T)
    ess_vec  = Vector{Float64}(undef, T)
    qn_vec   = Vector{Float64}(undef, T)

    for t in 1:T
        vt = vs[t]

        # ── Resample or identity ancestors ───────────────────────────────────
        ess = 1.0 / sum(w.^2)
        if ess < ess_min
            A         = systematic_resample(w, N, rng)
            w_hat     = ones(N)
            resampled = true
        else
            A         = collect(1:N)
            w_hat     = w_raw
            resampled = false
        end

        X_prev = X[A]

        # ── Propose X_t^n ~ P^{g_t+ε}(X_{t-1}^{A^n}, ·) ────────────────────
        X_new = [sample_twisted(X_prev[i], vt, ε, p, rng) for i in 1:N]

        # ── Incremental weight G_t(x_{t-1}, x_t) ─────────────────────────────
        G = [h_t(X_new[i], vt, p) / (g_t(X_new[i], vt, p) + ε) *
             (c_closed(X_prev[i], vt, p) + ε)
             for i in 1:N]

        # ── Update weights w_t^n = ŵ_{t-1}^n · G_t^n ────────────────────────
        w_raw_new = w_hat .* G

        # ── Likelihood increment ℓ_t^N (eq. 10.3) ────────────────────────────
        if resampled
            log_liks[t] = log(mean(w_raw_new))
        else
            log_liks[t] = log(sum(w_raw_new)) - log(sum(w_raw))
        end

        w_sum      = sum(w_raw_new)
        w          = w_raw_new ./ w_sum
        w_raw      = w_raw_new
        ess_vec[t] = 1.0 / sum(w.^2)
        qn_vec[t]  = maximum(w)

        X = X_new
    end

    return log_liks, ess_vec, minimum(ess_vec), qn_vec, maximum(qn_vec), X
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main_smc(p; T=50, N=500, ε=0.01, seed=42)
    rng = MersenneTwister(seed)

    xs_true, vs = simulate_hmm(0.0, T, p; rng=rng)
    println("Simulated $T observations.")

    log_liks, ess_vec, _, _, _, particles = run_smc(vs, N, ε, p; rng=rng)

    println("Total log-likelihood: $(sum(log_liks))")
    println("Mean ESS: $(round(mean(ess_vec), digits=1)) / $N")
    println("Min  ESS: $(round(minimum(ess_vec), digits=1)) / $N")

    ts = 1:T
    p1 = plot(ts, ess_vec,
              xlabel="t", ylabel="ESS",
              title="Effective sample size (N=$N, eps=$ε)",
              lw=2, legend=false)
    hline!(p1, [N/2], linestyle=:dash, color=:red)

    p2 = plot(ts, cumsum(log_liks),
              xlabel="t", ylabel="cumulative log-likelihood",
              title="Cumulative log p(v_{1:t})",
              lw=2, legend=false, color=:blue)

    savefig(p1, "figs/smc_ess.png")
    savefig(p2, "figs/smc_loglik.png")
    println("Plots saved.")

    return log_liks, ess_vec, particles
end

# ── Adaptive ε* SMC via grid search on m̂(ε) ─────────────────────────────────
function find_eps_star(x_prev, vt, p, rng, w;
                       ε_grid = exp.(range(log(1e-6), log(1.0), length=50)))
    @unpack ω, ψ, η = p
    N         = length(x_prev)
    proposals = [randn(rng) * η + ω + ψ * x_prev[i] for i in 1:N]
    h_vals    = [h_t(proposals[i], vt, p) for i in 1:N]
    g_vals    = [g_t(proposals[i], vt, p) for i in 1:N]
    c_vals    = [c_closed(x_prev[i], vt, p) for i in 1:N]
    # weighted average of per-particle m̂(ε)
    m_vals    = [sum(w[i] * (c_vals[i] + ε) * h_vals[i]^2 / (g_vals[i] + ε)
                     for i in 1:N)
                 for ε in ε_grid]
    return ε_grid[argmin(m_vals)]
end


function find_eps_star_optim(x_prev, vt, p, rng, w; ε_lo=1e-8, ε_hi=1.0)
    @unpack ω, ψ, η = p
    N         = length(x_prev)
    proposals = [randn(rng) * η + ω + ψ * x_prev[i] for i in 1:N]
    h_vals    = [h_t(proposals[i], vt, p) for i in 1:N]
    g_vals    = [g_t(proposals[i], vt, p) for i in 1:N]
    c_vals    = [c_closed(x_prev[i], vt, p) for i in 1:N]

    function m_hat(ε)
        return sum(w[i] * (c_vals[i] + ε) * h_vals[i]^2 / (g_vals[i] + ε)
                   for i in 1:N)
    end

    function dm_hat(ε)
        return sum(w[i] * h_vals[i]^2 * (g_vals[i] - c_vals[i]) / (g_vals[i] + ε)^2
                   for i in 1:N)
    end

    dm_hat(ε_lo) >= 0.0 && return ε_lo
    dm_hat(ε_hi) <= 0.0 && return ε_hi
    result = Optim.optimize(m_hat, ε_lo, ε_hi, Optim.Brent())
    return Optim.minimizer(result)
end

# """
#     find_eps_star(x_prev, vt, p, rng; ε_grid)

# Find ε* minimising the empirical m̂(ε) over a log-spaced grid, using
# auxiliary proposals drawn once from the untwisted P.
# """
# function find_eps_star(x_prev, vt, p, rng;
#                        ε_grid = exp.(range(log(1e-6), log(1.0), length=50)))
#     @unpack ω, ψ, η = p
#     N         = length(x_prev)
#     proposals = [randn(rng) * η + ω + ψ * x_prev[i] for i in 1:N]
#     h_vals    = [h_t(proposals[i], vt, p) for i in 1:N]
#     g_vals    = [g_t(proposals[i], vt, p) for i in 1:N]
#     c_mean    = mean(c_closed(x_prev[i], vt, p) for i in 1:N)
#     m_vals    = [(c_mean + ε) * mean(h_vals[i]^2 / (g_vals[i] + ε) for i in 1:N)
#                  for ε in ε_grid]
#     return ε_grid[argmin(m_vals)]
# end

# """
#     find_eps_star_optim(x_prev, vt, p, rng; ε_lo, ε_hi)

# Find ε* minimising m̂(ε) using Brent's method with analytical gradient:
#     dm̂/dε = I(ε) - (c̄ + ε) · J(ε)
# where I(ε) = (1/N)Σ hᵢ²/(gᵢ+ε),  J(ε) = (1/N)Σ hᵢ²/(gᵢ+ε)²
# """
# function find_eps_star_optim(x_prev, vt, p, rng;
#                              ε_lo = 1e-8,
#                              ε_hi = 1.0)
#     @unpack ω, ψ, η = p
#     N         = length(x_prev)
#     proposals = [randn(rng) * η + ω + ψ * x_prev[i] for i in 1:N]
#     h_vals    = [h_t(proposals[i], vt, p) for i in 1:N]
#     g_vals    = [g_t(proposals[i], vt, p) for i in 1:N]
#     c_mean    = mean(c_closed(x_prev[i], vt, p) for i in 1:N)

#     function m_hat(ε)
#         I = mean(h_vals[i]^2 / (g_vals[i] + ε) for i in 1:N)
#         return (c_mean + ε) * I
#     end

#     function dm_hat(ε)
#         I = mean(h_vals[i]^2 / (g_vals[i] + ε)   for i in 1:N)
#         J = mean(h_vals[i]^2 / (g_vals[i] + ε)^2 for i in 1:N)
#         return I - (c_mean + ε) * J
#     end

#     dm_hat(ε_lo) >= 0.0 && return ε_lo
#     dm_hat(ε_hi) <= 0.0 && return ε_hi

#     result = Optim.optimize(m_hat, ε_lo, ε_hi, Optim.Brent())
#     return Optim.minimizer(result)
# end

# ── Shared SMC loop body ──────────────────────────────────────────────────────

function _run_smc_adaptive(vs, N, p, find_eps_fn; rng=Random.default_rng(), ess_min=N/2)
    @unpack ψ, η = p
    T      = length(vs)
    σ_stat = η / sqrt(1 - ψ^2)

    X     = σ_stat * randn(rng, N)
    w     = fill(1.0/N, N)
    w_raw = fill(1.0,   N)

    log_liks = Vector{Float64}(undef, T)
    ess_vec  = Vector{Float64}(undef, T)
    qn_vec   = Vector{Float64}(undef, T)
    ε_vec    = Vector{Float64}(undef, T)

    for t in 1:T
        vt = vs[t]

        ε        = find_eps_fn(X, vt, p, rng, w)
        ε_vec[t] = ε

        ess = 1.0 / sum(w.^2)
        if ess < ess_min
            A         = systematic_resample(w, N, rng)
            w_hat     = ones(N)
            resampled = true
        else
            A         = collect(1:N)
            w_hat     = w_raw
            resampled = false
        end

        X_prev    = X[A]
        X_new     = [sample_twisted(X_prev[i], vt, ε, p, rng) for i in 1:N]
        G         = [h_t(X_new[i], vt, p) / (g_t(X_new[i], vt, p) + ε) *
                     (c_closed(X_prev[i], vt, p) + ε) for i in 1:N]
        w_raw_new = w_hat .* G

        if resampled
            log_liks[t] = log(mean(w_raw_new))
        else
            log_liks[t] = log(sum(w_raw_new)) - log(sum(w_raw))
        end

        w_sum      = sum(w_raw_new)
        w          = w_raw_new ./ w_sum
        w_raw      = w_raw_new
        ess_vec[t] = 1.0 / sum(w.^2)
        qn_vec[t]  = maximum(w)
        X          = X_new
    end

    return log_liks, ess_vec, minimum(ess_vec), qn_vec, maximum(qn_vec), ε_vec, X
end

"""GPF with ε_t* chosen by grid search on m̂(ε)."""
run_smc_adaptive(vs, N, p; rng=Random.default_rng(), ess_min=N/2) =
    _run_smc_adaptive(vs, N, p, find_eps_star; rng=rng, ess_min=ess_min)

"""GPF with ε_t* chosen by Brent optimisation on m̂(ε)."""
run_smc_adaptive_optim(vs, N, p; rng=Random.default_rng(), ess_min=N/2) =
    _run_smc_adaptive(vs, N, p, find_eps_star_optim; rng=rng, ess_min=ess_min)

# ── Compare fixed vs adaptive ─────────────────────────────────────────────────


"""
Compare a range of fixed ε values, ε=0, and both adaptive methods (grid
and Brent) for both regimes. Produces one CDF plot of min-ESS per regime,
showing all cases on the same axes.
"""
function compare_adaptive(;
        T        = 50,
        N        = 500,
        R        = 100,
        η_good   = 1.0,
        η_bad    = 3.0,
        ε_range  = [0.0, 0.0001, 0.001, 0.01, 0.1],
        seed     = 42)

    # Colours for fixed ε: gradient from light to dark
    ε_colors = [:lightcoral, :orange, :goldenrod, :olivedrab, :teal]

    for (label, η_val, slug) in [("η=$(η_good) < σ", η_good, "good"),
                                  ("η=$(η_bad) > σ",  η_bad,  "bad")]

        p = Para(η=η_val)

        rng_data = MersenneTwister(seed)
        _, vs    = simulate_hmm(0.0, T, p; rng=rng_data)

        # Collect min-ESS and max-Qn for each fixed ε
        miness_fixed = Dict{Float64, Vector{Float64}}(ε => Float64[] for ε in ε_range)
        maxqn_fixed  = Dict{Float64, Vector{Float64}}(ε => Float64[] for ε in ε_range)
        miness_grid  = Float64[]
        miness_optim = Float64[]
        maxqn_grid   = Float64[]
        maxqn_optim  = Float64[]
        eps_grid     = Vector{Vector{Float64}}()
        eps_optim    = Vector{Vector{Float64}}()

        for r in 1:R
            for ε in ε_range
                rng_r = MersenneTwister(seed + r)
                _, _, min_ess_f, _, max_qn_f, _ = run_smc(vs, N, ε, p; rng=rng_r)
                push!(miness_fixed[ε], min_ess_f)
                push!(maxqn_fixed[ε],  max_qn_f)
            end

            rng_r2 = MersenneTwister(seed + r)
            _, _, min_ess_g, _, max_qn_g, ε_vec_g, _ = run_smc_adaptive(vs, N, p; rng=rng_r2)
            push!(miness_grid, min_ess_g)
            push!(maxqn_grid,  max_qn_g)
            push!(eps_grid, ε_vec_g)

            rng_r3 = MersenneTwister(seed + r)
            _, _, min_ess_o, _, max_qn_o, ε_vec_o, _ = run_smc_adaptive_optim(vs, N, p; rng=rng_r3)
            push!(miness_optim, min_ess_o)
            push!(maxqn_optim,  max_qn_o)
            push!(eps_optim, ε_vec_o)
        end

        # Print summary
        println("$label:")
        for ε in ε_range
            me = miness_fixed[ε]
            println("  Fixed ϵ=$ε : mean=$(round(mean(me),digits=1)), "*
                    "5th pct=$(round(quantile(me,0.05),digits=1))")
        end
        println("  Adaptive grid  : mean=$(round(mean(miness_grid),digits=1)), "*
                "5th pct=$(round(quantile(miness_grid,0.05),digits=1))")
        println("  Adaptive Brent : mean=$(round(mean(miness_optim),digits=1)), "*
                "5th pct=$(round(quantile(miness_optim,0.05),digits=1))")

        # CDF plot
        plt = plot(title=label, xlabel="min ESS", ylabel="P(min ESS <= x)",
                   size=(650, 400))
        for (k, ε) in enumerate(ε_range)
            vals = sort(miness_fixed[ε])
            cdf  = (1:R) ./ R
            plot!(plt, vals, cdf, label="ϵ=$ε", lw=2,
                  color=ε_colors[min(k, length(ε_colors))])
        end
        let vals = sort(miness_grid), cdf = (1:R) ./ R
            plot!(plt, vals, cdf, label="adaptive (grid)",
                  lw=2, color=:blue, linestyle=:dash)
        end
        let vals = sort(miness_optim), cdf = (1:R) ./ R
            plot!(plt, vals, cdf, label="adaptive (Brent)",
                  lw=2, color=:purple, linestyle=:dot)
        end
        savefig(plt, "figs/adaptive_miness_cdf_$slug.png")

        # max-Qn CDF plot (Chatterjee-Diaconis diagnostic)
        # Large Qn is bad (weight concentration), so curves to the LEFT are worse
        plt_qn = plot(title=label, xlabel="max Qn", ylabel="P(max Qn <= x)",
                      size=(650, 400))
        for (k, ε) in enumerate(ε_range)
            vals = sort(maxqn_fixed[ε])
            cdf  = (1:R) ./ R
            plot!(plt_qn, vals, cdf, label="ϵ=$ε", lw=2,
                  color=ε_colors[min(k, length(ε_colors))])
        end
        let vals = sort(maxqn_grid), cdf = (1:R) ./ R
            plot!(plt_qn, vals, cdf, label="adaptive (grid)",
                  lw=2, color=:blue, linestyle=:dash)
        end
        let vals = sort(maxqn_optim), cdf = (1:R) ./ R
            plot!(plt_qn, vals, cdf, label="adaptive (Brent)",
                  lw=2, color=:purple, linestyle=:dot)
        end
        savefig(plt_qn, "figs/adaptive_maxqn_cdf_$slug.png")

        # ε* over time
        p2 = plot(title="Mean ϵ* over time ($label)",
                  xlabel="t", ylabel="ϵ*",
                  yscale=:log10, lw=2, size=(600, 350))
        plot!(p2, 1:T, mean(eps_grid),  label="grid",  lw=2, color=:blue)
        plot!(p2, 1:T, mean(eps_optim), label="Brent", lw=2, color=:purple,
              linestyle=:dash)
        savefig(p2, "figs/adaptive_eps_over_time_$slug.png")
    end

    println("\nSaved: adaptive_miness_cdf_good/bad.png, adaptive_maxqn_cdf_good/bad.png, adaptive_eps_over_time_good/bad.png")
end

# ── Speed comparison ──────────────────────────────────────────────────────────

function benchmark_eps_methods(p; N=500, T=50, seed=42)
    @unpack ψ, η = p
    rng      = MersenneTwister(seed)
    _, vs    = simulate_hmm(0.0, T, p; rng=rng)
    rng_data = MersenneTwister(seed + 1)
    X        = (η / sqrt(1 - ψ^2)) * randn(rng_data, N)

    # Warmup
    find_eps_star(X, vs[1], p, MersenneTwister(1))
    find_eps_star_optim(X, vs[1], p, MersenneTwister(1))

    t_grid  = @elapsed for t in 1:T
        find_eps_star(X, vs[t], p, MersenneTwister(t))
    end
    t_brent = @elapsed for t in 1:T
        find_eps_star_optim(X, vs[t], p, MersenneTwister(t))
    end

    println("Timing over $T steps (N=$N):")
    println("  Grid search : $(round(t_grid*1000,  digits=2)) ms total, "*
            "$(round(t_grid/T*1000,  digits=3)) ms/step")
    println("  Brent       : $(round(t_brent*1000, digits=2)) ms total, "*
            "$(round(t_brent/T*1000, digits=3)) ms/step")
    println("  Speedup     : $(round(t_grid/t_brent, digits=2))x in favour of "*
            (t_grid > t_brent ? "Brent" : "grid search"))
end

# ── Run ───────────────────────────────────────────────────────────────────────

p = Para()
main_smc(p)
compare_adaptive(R=500)
benchmark_eps_methods(Para())



# ── Single-run visualisation ──────────────────────────────────────────────────
 
"""
    plot_smc_run(vs, xs_true, log_liks, ess_vec, slug; ε_vec=nothing)
 
Visualise the results of a single SMC run with four panels:
  1. ESS over time
  2. Cumulative log-likelihood over time
  3. Incremental log-likelihood over time
  4. ε* over time (only shown if ε_vec is provided, i.e. adaptive run)
 
Saves to smc_run_<slug>.png.
"""
function plot_smc_run(vs, xs_true, log_liks, ess_vec, slug; ε_vec=nothing)
    T  = length(vs)
    ts = 1:T
 
    n_panels = isnothing(ε_vec) ? 3 : 4
    plts     = []
 
    # Panel 1: ESS
    p1 = plot(ts, ess_vec,
              xlabel="t", ylabel="ESS",
              title="ESS over time",
              lw=2, legend=false, color=:blue)
    push!(plts, p1)
 
    # Panel 2: cumulative log-likelihood
    p2 = plot(ts, cumsum(log_liks),
              xlabel="t", ylabel="cumulative log-lik",
              title="Cumulative log p(v_{1:t})",
              lw=2, legend=false, color=:black)
    push!(plts, p2)
 
    # Panel 3: incremental log-likelihood
    p3 = plot(ts, log_liks,
              xlabel="t", ylabel="log lik increment",
              title="Incremental log p(v_t | v_{1:t-1})",
              lw=2, legend=false, color=:green)
    push!(plts, p3)
 
    # Panel 4: ε* over time (adaptive only)
    if !isnothing(ε_vec)
        p4 = plot(ts, ε_vec,
                  xlabel="t", ylabel="eps*",
                  title="Adaptive eps* over time",
                  lw=2, legend=false, color=:orange,
                  yscale=:log10)
        push!(plts, p4)
    end
 
    layout = n_panels == 3 ? (1, 3) : (2, 2)
    plt = plot(plts..., layout=layout, size=(900, n_panels == 3 ? 300 : 600))
    savefig(plt, "figs/smc_run_$slug.png")
    println("Saved smc_run_$slug.png")
    return plt
end
 
"""
    run_and_plot(p; T, N, ε, seed, adaptive)
 
Run SMC once and visualise. If adaptive=true, uses grid-search adaptive ε*;
otherwise uses the fixed ε provided.
"""
function run_and_plot(p; T=50, N=500, ε=0.01, seed=42, adaptive=false)
    rng = MersenneTwister(seed)
    xs_true, vs = simulate_hmm(0.0, T, p; rng=rng)
 
    @unpack η = p
    slug = adaptive ? "adaptive_eta$(η)" : "fixed_eps$(ε)_eta$(η)"
 
    if adaptive
        log_liks, ess_vec, _, _, _, ε_vec, _ = run_smc_adaptive(vs, N, p; rng=rng)
        plot_smc_run(vs, xs_true, log_liks, ess_vec, slug; ε_vec=ε_vec)
    else
        log_liks, ess_vec, _, _, _, _ = run_smc(vs, N, ε, p; rng=rng)
        plot_smc_run(vs, xs_true, log_liks, ess_vec, slug)
    end
 
    println("Total log-likelihood : $(round(sum(log_liks), digits=2))")
    println("Mean ESS             : $(round(mean(ess_vec), digits=1)) / $N")
    println("Min  ESS             : $(round(minimum(ess_vec), digits=1)) / $N")
end
 
# Example calls
run_and_plot(Para(η=1.0); adaptive=false, ε=0.0)
run_and_plot(Para(η=1.0); adaptive=true)
run_and_plot(Para(η=3.0); adaptive=false, ε=0.0)
run_and_plot(Para(η=3.0); adaptive=true)