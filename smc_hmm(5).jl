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
    c(x)      = N(vt - μ - ω - ψx; 0, η² + σ²)        [closed form]

IS weight:
    G_ε(x, x₁) = h(x₁) · (c(x) + ε) / (g(x₁) + ε)
"""

include("onestep_IS.jl")   # loads ω, ψ, η, μ, σ², v, h(x₁)

using Random
using Statistics

# ── Closed-form expressions ───────────────────────────────────────────────────

"""
c(x, vt) = ∫ P(x, dx₁) g_t(x₁)  =  N(vt - μ; ω + ψx, η² + σ²)
"""
c_closed(x, vt) = pdf(Normal(ω + ψ*x, sqrt(η^2 + σ²)), vt - μ)

"""
Parameters of P^g(x, ·) = N(·; μ*, σ*):
  1/σ*² = 1/η² + 1/σ²
  μ*    = σ*² · ((ω + ψx)/η² + (vt - μ)/σ²)
"""
function Pg_params(x, vt)
    σ²_star = 1.0 / (1.0/η^2 + 1.0/σ²)
    μ_star  = σ²_star * ((ω + ψ*x)/η^2 + (vt - μ)/σ²)
    return μ_star, sqrt(σ²_star)
end

"""g_t(x₁, vt) = N(x₁; vt - μ, σ²)"""
g_t(x₁, vt) = pdf(Normal(vt - μ, sqrt(σ²)), x₁)

"""h_t(x₁, vt) = f_{χ²(1)}(e^{vt-x₁}) · e^{vt-x₁}"""
function h_t(x₁, vt)
    u = exp(vt - x₁)
    return pdf(Chisq(1), u) * u
end

"""
Sample from P^{g+ε}(x, ·):
  with prob α = c(x)/(c(x)+ε): draw from P^g  = N(μ*, σ*)
  with prob 1-α:                draw from P    = N(ω+ψx, η)
"""
function sample_twisted(x, vt, ε, rng)
    ct = c_closed(x, vt)
    α  = ct / (ct + ε)
    if rand(rng) < α
        μ_star, σ_star = Pg_params(x, vt)
        return randn(rng) * σ_star + μ_star
    else
        return randn(rng) * η + ω + ψ*x
    end
end

# ── Simulate HMM ─────────────────────────────────────────────────────────────

function simulate_hmm(x0, T; rng=Random.default_rng())
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
    u = rand(rng) / N
    idx = Vector{Int}(undef, N)
    cumw = cumsum(weights)
    j = 1
    for i in 1:N
        target = u + (i-1)/N
        while j < N && cumw[j] < target
            j += 1
        end
        idx[i] = j
    end
    return idx
end

# ── SMC ───────────────────────────────────────────────────────────────────────

"""
Run SMC with N particles for observations vs = (v_1,...,v_T).

Returns:
  log_liks : incremental log p̂(v_t | v_{1:t-1})
  ess_vec  : ESS at each step (before resampling)
  particles: final particle cloud
"""
function run_smc(vs, N, ε; rng=Random.default_rng())
    T = length(vs)

    # Initialise from stationary distribution N(0, η²/(1-ψ²))
    σ_stat    = η / sqrt(1 - ψ^2)
    particles = σ_stat * randn(rng, N)

    log_liks  = Vector{Float64}(undef, T)
    ess_vec   = Vector{Float64}(undef, T)

    for t in 1:T
        vt   = vs[t]
        prev = copy(particles)

        # Propose x_t ~ P^{g_t+ε}(x_{t-1}, ·)  [exact mixture sampler]
        for i in 1:N
            particles[i] = sample_twisted(prev[i], vt, ε, rng)
        end

        # Unnormalised weights  w_i = h_t(x_i) / (g_t(x_i) + ε)
        raw_w = [h_t(particles[i], vt) / (g_t(particles[i], vt) + ε)
                 for i in 1:N]

        # Incremental log-likelihood:  log[ (c̄ + ε) · mean(w) ]
        ct_mean     = mean(c_closed(prev[i], vt) for i in 1:N)
        log_liks[t] = log(ct_mean + ε) + log(mean(raw_w))

        # Normalise and compute ESS
        norm_w     = raw_w ./ sum(raw_w)
        ess_vec[t] = 1.0 / sum(norm_w.^2)

        # Resample if ESS < N/2
        if ess_vec[t] < N/2
            idx        = systematic_resample(norm_w, N, rng)
            particles  = particles[idx]
        end
    end

    return log_liks, ess_vec, minimum(ess_vec), particles
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main_smc(; T=50, N=500, ε=0.01, seed=42)
    rng = MersenneTwister(seed)

    xs_true, vs = simulate_hmm(0.0, T; rng=rng)
    println("Simulated $T observations.")

    log_liks, ess_vec, _, particles = run_smc(vs, N, ε; rng=rng)

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

    savefig(p1, "smc_ess.png")
    savefig(p2, "smc_loglik.png")
    println("Plots saved.")

    return log_liks, ess_vec, minimum(ess_vec), particles
end

main_smc()

# ── Comparison experiment ─────────────────────────────────────────────────────

"""
compare_epsilon(; ...)

Runs two experiments:
  Experiment A — finite-variance regime:   η < σ  (η = 1.0)
  Experiment B — infinite-variance regime: η > σ  (η = 3.0)

For each regime, SMC is run for several values of ε. ESS and cumulative
log-likelihood are plotted. The collapse of ESS at ε=0 in regime B is the
key illustration of why ε matters.
"""
function compare_epsilon(;
        T      = 100,
        N      = 500,
        εs     = [0.0, 0.001, 0.01, 0.1],
        η_good = 1.0,    # η < σ ≈ 2.22  →  finite variance at ε=0
        η_bad  = 3.0,    # η > σ          →  infinite variance at ε=0
        seed   = 42)

    results = Dict()

    for (label, η_val) in [("A: eta=$(η_good) < sigma", η_good),
                            ("B: eta=$(η_bad) > sigma",  η_bad)]

        global η = η_val

        # Simulate data once; same dataset for all ε
        rng  = MersenneTwister(seed)
        _, vs = simulate_hmm(0.0, T; rng=rng)

        ess_runs    = Dict()
        loglik_runs = Dict()

        for ε in εs
            rng2 = MersenneTwister(seed + 1)
            log_liks, ess_vec, _, _ = run_smc(vs, N, ε; rng=rng2)
            ess_runs[ε]    = ess_vec
            loglik_runs[ε] = cumsum(log_liks)
            println("  $label, ε=$ε : "*
                    "mean ESS=$(round(mean(ess_vec),digits=1)), "*
                    "min ESS=$(round(minimum(ess_vec),digits=1)), "*
                    "total loglik=$(round(sum(log_liks),digits=2))")
        end

        results[label] = (ess_runs, loglik_runs)
    end

    # Reset η to default
    global η = 1.0

    ts = 1:T

    for (label, η_val) in [("A: eta=$(η_good) < sigma", η_good),
                            ("B: eta=$(η_bad) > sigma",  η_bad)]

        ess_runs, loglik_runs = results[label]
        slug = η_val == η_good ? "good" : "bad"

        # ESS plot
        p1 = plot(title=label, xlabel="t", ylabel="ESS",
                  ylims=(0, N*1.05), lw=2, size=(600,350))
        for ε in εs
            plot!(p1, ts, ess_runs[ε], label="eps=$ε", lw=2)
        end
        hline!(p1, [N/2], linestyle=:dash, color=:black, label="N/2")
        savefig(p1, "compare_ess_$slug.png")

        # Cumulative log-likelihood plot
        p2 = plot(title=label, xlabel="t", ylabel="cumulative log-lik",
                  lw=2, size=(600,350))
        for ε in εs
            plot!(p2, ts, loglik_runs[ε], label="eps=$ε", lw=2)
        end
        savefig(p2, "compare_loglik_$slug.png")
    end

    println("\nSaved: compare_ess_good.png, compare_ess_bad.png, "*
            "compare_loglik_good.png, compare_loglik_bad.png")
    return results
end

compare_epsilon()


# ── Min-ESS across runs experiment ───────────────────────────────────────────

"""
miness_across_runs(; ...)

For each combination of (η, ε), runs SMC R independent times on the same
observed dataset and records the minimum ESS over time for each run.

The key prediction from theory:
  - η < σ: min-ESS distribution is similar across all ε
  - η > σ: at ε=0, min-ESS occasionally collapses to near 1 (a single
            particle dominates), while positive ε prevents this.

Produces:
  - For each regime: empirical CDF of min-ESS across runs, one curve per ε.
    A curve concentrated near 0 indicates frequent weight collapse.
  - Mean and 5th percentile of min-ESS as a function of ε, for both regimes.
"""
function miness_across_runs(;
        T      = 50,
        N      = 200,
        R      = 200,
        εs     = [0.0, 0.001, 0.01, 0.05, 0.1],
        η_good = 1.0,
        η_bad  = 3.0,
        seed   = 42)

    all_results = Dict()

    for (label, η_val, slug) in [("eta=$(η_good) < sigma", η_good, "good"),
                                  ("eta=$(η_bad) > sigma",  η_bad,  "bad")]

        global η = η_val

        # Same dataset for all ε and all runs
        rng_data = MersenneTwister(seed)
        _, vs    = simulate_hmm(0.0, T; rng=rng_data)

        miness_per_ε = Dict()   # ε => vector of R min-ESS values

        for ε in εs
            min_ess_runs = Float64[]
            for r in 1:R
                rng_r = MersenneTwister(seed + r)
                _, _, min_ess, _ = run_smc(vs, N, ε; rng=rng_r)
                push!(min_ess_runs, min_ess)
            end
            miness_per_ε[ε] = min_ess_runs
            println("  $label, ε=$ε: "*
                    "mean min-ESS=$(round(mean(min_ess_runs),digits=1)), "*
                    "5th pct=$(round(quantile(min_ess_runs,0.05),digits=1))")
        end

        all_results[slug] = (label, miness_per_ε)
    end

    global η = 1.0

    # ── Plot 1: empirical CDF of min-ESS for each regime ─────────────────────
    for (slug, color_cycle) in [("good", [:blue, :cyan, :teal, :navy, :dodgerblue]),
                                 ("bad",  [:red, :orange, :coral, :firebrick, :salmon])]
        label, miness_per_ε = all_results[slug]
        p = plot(title=label, xlabel="min ESS", ylabel="P(min ESS <= x)",
                 xlims=(0, N), size=(600, 350))
        for (i, ε) in enumerate(εs)
            vals = sort(miness_per_ε[ε])
            cdf  = (1:R) ./ R
            plot!(p, vals, cdf, label="eps=$ε", lw=2,
                  color=color_cycle[min(i, length(color_cycle))])
        end
        vline!(p, [N/2], linestyle=:dash, color=:black, label="N/2")
        savefig(p, "miness_cdf_$slug.png")
    end

    # ── Plot 2: mean and 5th percentile of min-ESS vs ε ──────────────────────
    p2 = plot(xlabel="epsilon", ylabel="min ESS",
              title="Min ESS across $R runs (mean and 5th percentile)",
              size=(600, 350))
    for (slug, color) in [("good", :blue), ("bad", :red)]
        label, miness_per_ε = all_results[slug]
        means = [mean(miness_per_ε[ε])          for ε in εs]
        p05   = [quantile(miness_per_ε[ε], 0.05) for ε in εs]
        plot!(p2, εs, means, label="$label mean", lw=2,
              marker=:circle, color=color)
        plot!(p2, εs, p05,   label="$label 5th pct", lw=2,
              marker=:diamond, linestyle=:dash, color=color)
    end
    hline!(p2, [N/2], linestyle=:dot, color=:black, label="N/2")
    savefig(p2, "miness_vs_epsilon.png")

    println("\nSaved: miness_cdf_good.png, miness_cdf_bad.png, miness_vs_epsilon.png")
    return all_results
end

miness_across_runs()

# ── Adaptive ε* SMC ───────────────────────────────────────────────────────────

"""
    find_eps_star(x_prev_particles, vt; ε_grid)

Given the current particle cloud x_prev_particles at time t-1 and the
observation vt, find the ε* that minimises the empirical estimate of m(ε):

    m̂(ε) = (c̄ + ε) · (1/N) Σᵢ h(xᵢ)² / (g(xᵢ) + ε)

where xᵢ are proposed from P(x_prev, ·) and c̄ = mean c(x_prev).

We minimise over a log-spaced grid. The proposals are drawn once and reused
across all ε values, so the cost is one forward pass plus a grid search.
"""
function find_eps_star(x_prev, vt, rng;
                       ε_grid = exp.(range(log(1e-6), log(1.0), length=50)))

    N = length(x_prev)

    # Draw proposals from untwisted P(x_{t-1}, ·) once
    proposals = [randn(rng) * η + ω + ψ * x_prev[i] for i in 1:N]

    # Precompute h and g at proposals
    h_vals = [h_t(proposals[i], vt) for i in 1:N]
    g_vals = [g_t(proposals[i], vt) for i in 1:N]
    c_vals = [c_closed(x_prev[i], vt) for i in 1:N]
    c_mean = mean(c_vals)

    # Evaluate m̂(ε) over grid
    m_vals = [(c_mean + ε) * mean(h_vals[i]^2 / (g_vals[i] + ε) for i in 1:N)
              for ε in ε_grid]

    ε_star = ε_grid[argmin(m_vals)]
    return ε_star
end

"""
    run_smc_adaptive(vs, N; ε_grid, rng)

SMC with adaptive ε*_t chosen at each step by minimising the empirical m̂(ε)
over the current particle cloud. Everything else is identical to run_smc.

Returns log_liks, ess_vec, min_ess, ε_vec (the chosen ε at each step),
and the final particles.
"""
function run_smc_adaptive(vs, N;
                          ε_grid = exp.(range(log(1e-6), log(1.0), length=50)),
                          rng    = Random.default_rng())

    T      = length(vs)
    σ_stat = η / sqrt(1 - ψ^2)
    particles = σ_stat * randn(rng, N)

    log_liks = Vector{Float64}(undef, T)
    ess_vec  = Vector{Float64}(undef, T)
    ε_vec    = Vector{Float64}(undef, T)

    for t in 1:T
        vt   = vs[t]
        prev = copy(particles)

        # ── Find ε* for this step ─────────────────────────────────────────────
        ε = find_eps_star(prev, vt, rng; ε_grid=ε_grid)
        ε_vec[t] = ε

        # ── Propose from P^{g+ε*} ─────────────────────────────────────────────
        for i in 1:N
            particles[i] = sample_twisted(prev[i], vt, ε, rng)
        end

        # ── Weights, likelihood, ESS, resample ───────────────────────────────
        raw_w = [h_t(particles[i], vt) / (g_t(particles[i], vt) + ε)
                 for i in 1:N]

        ct_mean     = mean(c_closed(prev[i], vt) for i in 1:N)
        log_liks[t] = log(ct_mean + ε) + log(mean(raw_w))

        norm_w     = raw_w ./ sum(raw_w)
        ess_vec[t] = 1.0 / sum(norm_w.^2)

        if ess_vec[t] < N/2
            idx       = systematic_resample(norm_w, N, rng)
            particles = particles[idx]
        end
    end

    return log_liks, ess_vec, minimum(ess_vec), ε_vec, particles
end

"""
Compare fixed-ε SMC against adaptive-ε* SMC for both regimes.
"""
function compare_adaptive(;
        T      = 50,
        N      = 500,
        R      = 100,
        η_good = 1.0,
        η_bad  = 3.0,
        ε_fix  = 0.0,     # fixed ε to compare against (use 0 for worst case)
        seed   = 42)

    for (label, η_val, slug) in [("eta=$(η_good) < sigma", η_good, "good"),
                                  ("eta=$(η_bad) > sigma",  η_bad,  "bad")]

        global η = η_val

        rng_data = MersenneTwister(seed)
        _, vs    = simulate_hmm(0.0, T; rng=rng_data)

        miness_fixed    = Float64[]
        miness_adaptive = Float64[]
        eps_chosen      = Vector{Vector{Float64}}()

        for r in 1:R
            # Fixed ε
            rng_r = MersenneTwister(seed + r)
            _, _, min_ess_f, _ = run_smc(vs, N, ε_fix; rng=rng_r)
            push!(miness_fixed, min_ess_f)

            # Adaptive ε*
            rng_r2 = MersenneTwister(seed + r)
            _, _, min_ess_a, ε_vec, _ = run_smc_adaptive(vs, N; rng=rng_r2)
            push!(miness_adaptive, min_ess_a)
            push!(eps_chosen, ε_vec)
        end

        println("$label:")
        println("  Fixed ε=$ε_fix : mean min-ESS=$(round(mean(miness_fixed),digits=1)), "*
                "5th pct=$(round(quantile(miness_fixed,0.05),digits=1))")
        println("  Adaptive ε*   : mean min-ESS=$(round(mean(miness_adaptive),digits=1)), "*
                "5th pct=$(round(quantile(miness_adaptive,0.05),digits=1))")

        # CDF comparison
        p1 = plot(title=label, xlabel="min ESS", ylabel="P(min ESS <= x)",
                  xlims=(0, N), size=(600,350))
        let vals = sort(miness_fixed), cdf = (1:R)./R
            plot!(p1, vals, cdf, label="fixed eps=$ε_fix", lw=2, color=:red)
        end
        let vals = sort(miness_adaptive), cdf = (1:R)./R
            plot!(p1, vals, cdf, label="adaptive eps*", lw=2, color=:blue)
        end
        vline!(p1, [N/2], linestyle=:dash, color=:black, label="N/2")
        savefig(p1, "adaptive_miness_cdf_$slug.png")

        # Mean ε* chosen over time (averaged across runs)
        mean_eps = mean(eps_chosen)
        p2 = plot(1:T, mean_eps,
                  xlabel="t", ylabel="eps*",
                  title="Mean adaptive eps* over time ($label)",
                  lw=2, legend=false, yscale=:log10, size=(600,350))
        savefig(p2, "adaptive_eps_over_time_$slug.png")
    end

    global η = 1.0
    println("\nSaved: adaptive_miness_cdf_good/bad.png, adaptive_eps_over_time_good/bad.png")
end

compare_adaptive()

