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
using RCall

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

    ts     = collect(1:T)
    cum_ll = cumsum(log_liks)
    eps_val = ε

    @rput ts ess_vec cum_ll N eps_val

    R"""
    library(ggplot2)

    df_ess <- data.frame(t = ts, ess = ess_vec)
    p_ess <- ggplot(df_ess, aes(x = t, y = ess)) +
      geom_line(linewidth = 0.8, colour = "steelblue") +
      geom_hline(yintercept = N / 2, linetype = "dashed", colour = "red") +
      labs(x = "t", y = "ESS",
           title = paste0("Effective sample size (N=", N, ", eps=", eps_val, ")")) +
      theme_bw(base_size = 12)

    df_ll <- data.frame(t = ts, cum_ll = cum_ll)
    p_ll <- ggplot(df_ll, aes(x = t, y = cum_ll)) +
      geom_line(linewidth = 0.8, colour = "blue") +
      labs(x = "t", y = "cumulative log-likelihood",
           title = "Cumulative log p(v[1:t])") +
      theme_bw(base_size = 12)

    ggsave("figs/smc_ess.pdf", p_ess, width = 6, height = 4)
    ggsave("figs/smc_loglik.pdf", p_ll, width = 6, height = 4)
    """
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
and Brent) for both regimes (η_good, η_bad). Produces faceted ggplot figures
(one facet per η) via RCall: min-ESS CDF, max-Qn CDF, and mean ε* over time.
"""
function compare_adaptive(;
        T        = 50,
        N        = 500,
        R        = 100,
        η_good   = 1.0,
        η_bad    = 3.0,
        ε_range  = [0.0, 0.0001, 0.001, 0.01, 0.1],
        seed     = 42)

    # Long-format accumulators across both η regimes
    ess_rows = NamedTuple[]   # (eta_label, method, value)
    qn_rows  = NamedTuple[]   # (eta_label, method, value)
    eps_rows = NamedTuple[]   # (eta_label, method, t, eps_star)

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

        # ── Accumulate long-format rows for faceted plotting ─────────────────
        for ε in ε_range
            for v in miness_fixed[ε]
                push!(ess_rows, (eta_label=label, method="eps=$ε", value=v))
            end
            for v in maxqn_fixed[ε]
                push!(qn_rows, (eta_label=label, method="eps=$ε", value=v))
            end
        end
        for v in miness_grid
            push!(ess_rows, (eta_label=label, method="adaptive (grid)", value=v))
        end
        for v in miness_optim
            push!(ess_rows, (eta_label=label, method="adaptive (Brent)", value=v))
        end
        for v in maxqn_grid
            push!(qn_rows, (eta_label=label, method="adaptive (grid)", value=v))
        end
        for v in maxqn_optim
            push!(qn_rows, (eta_label=label, method="adaptive (Brent)", value=v))
        end

        eps_grid_mean  = mean(eps_grid)
        eps_optim_mean = mean(eps_optim)
        for t in 1:T
            push!(eps_rows, (eta_label=label, method="grid",  t=t, eps_star=eps_grid_mean[t]))
            push!(eps_rows, (eta_label=label, method="Brent", t=t, eps_star=eps_optim_mean[t]))
        end
    end

    # ── Flatten to plain vectors for RCall ────────────────────────────────────
    ess_eta    = [r.eta_label for r in ess_rows]
    ess_method = [r.method    for r in ess_rows]
    ess_value  = [r.value     for r in ess_rows]

    qn_eta    = [r.eta_label for r in qn_rows]
    qn_method = [r.method    for r in qn_rows]
    qn_value  = [r.value     for r in qn_rows]

    eps_eta    = [r.eta_label for r in eps_rows]
    eps_method = [r.method    for r in eps_rows]
    eps_t      = [r.t         for r in eps_rows]
    eps_star_v = [r.eps_star  for r in eps_rows]

    @rput ess_eta ess_method ess_value
    @rput qn_eta qn_method qn_value
    @rput eps_eta eps_method eps_t eps_star_v

    R"""
    library(ggplot2)
    library(dplyr)

    # ── min-ESS CDF, faceted by eta ──────────────────────────────────────────
    df_ess <- data.frame(eta = ess_eta, method = ess_method, value = ess_value) %>%
      group_by(eta, method) %>%
      arrange(value, .by_group = TRUE) %>%
      mutate(cdf = row_number() / n()) %>%
      ungroup()

    p_ess <- ggplot(df_ess, aes(x = value, y = cdf, colour = method, linetype = method)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ eta, scales = "free_x") +
      labs(x = "min ESS", y = "P(min ESS <= x)", colour = "Method", linetype = "Method") +
      theme_bw(base_size = 12) +
      theme(strip.background = element_blank(), strip.text = element_text(size = 12))

    ggsave("figs/adaptive_miness_cdf.pdf", p_ess, width = 9, height = 4)

    # ── max-Qn CDF, faceted by eta (Chatterjee-Diaconis diagnostic) ──────────
    # Large Qn is bad (weight concentration), so curves to the LEFT are worse
    df_qn <- data.frame(eta = qn_eta, method = qn_method, value = qn_value) %>%
      group_by(eta, method) %>%
      arrange(value, .by_group = TRUE) %>%
      mutate(cdf = row_number() / n()) %>%
      ungroup()

    p_qn <- ggplot(df_qn, aes(x = value, y = cdf, colour = method, linetype = method)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ eta, scales = "free_x") +
      labs(x = "max Qn", y = "P(max Qn <= x)", colour = "Method", linetype = "Method") +
      theme_bw(base_size = 12) +
      theme(strip.background = element_blank(), strip.text = element_text(size = 12))

    ggsave("figs/adaptive_maxqn_cdf.pdf", p_qn, width = 9, height = 4)

    # ── Mean eps* over time, faceted by eta ──────────────────────────────────
    df_eps <- data.frame(eta = eps_eta, method = eps_method, t = eps_t, eps_star = eps_star_v)

    p_eps <- ggplot(df_eps, aes(x = t, y = eps_star, colour = method, linetype = method)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ eta) +
      scale_y_log10() +
      labs(x = "t", y = expression(epsilon^"*"), colour = "Method", linetype = "Method",
           title = "Mean adaptive eps* over time") +
      theme_bw(base_size = 12) +
      theme(strip.background = element_blank(), strip.text = element_text(size = 12))

    ggsave("figs/adaptive_eps_over_time.pdf", p_eps, width = 9, height = 4)
    """

    println("\nSaved: figs/adaptive_miness_cdf.pdf, figs/adaptive_maxqn_cdf.pdf, "*
            "figs/adaptive_eps_over_time.pdf")
end

# ── Speed comparison ──────────────────────────────────────────────────────────

function benchmark_eps_methods(p; N=500, T=50, seed=42)
    @unpack ψ, η = p
    rng      = MersenneTwister(seed)
    _, vs    = simulate_hmm(0.0, T, p; rng=rng)
    rng_data = MersenneTwister(seed + 1)
    X        = (η / sqrt(1 - ψ^2)) * randn(rng_data, N)
    w        = fill(1.0/N, N)   # uniform weights for benchmarking
    # Warmup
    find_eps_star(X, vs[1], p, MersenneTwister(1), w)
    find_eps_star_optim(X, vs[1], p, MersenneTwister(1), w)

    t_grid  = @elapsed for t in 1:T
        find_eps_star(X, vs[t], p, MersenneTwister(t), w)
    end
    t_brent = @elapsed for t in 1:T
        find_eps_star_optim(X, vs[t], p, MersenneTwister(t), w)
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
compare_adaptive(R=500)#, η_bad    = 8.0)
benchmark_eps_methods(Para())



# ── Single-run visualisation ──────────────────────────────────────────────────

"""
    run_smc_result(p; T, N, ε, seed, adaptive)

Run SMC once and return the raw time series needed for plotting (no plotting
here — see `plot_smc_runs_faceted`). If adaptive=true, uses grid-search
adaptive ε*; otherwise uses the fixed ε provided.
"""
function run_smc_result(p; T=50, N=500, ε=0.01, seed=42, adaptive=false)
    rng = MersenneTwister(seed)
    _, vs = simulate_hmm(0.0, T, p; rng=rng)

    if adaptive
        log_liks, ess_vec, _, _, _, ε_vec, _ = run_smc_adaptive(vs, N, p; rng=rng)
    else
        log_liks, ess_vec, _, _, _, _ = run_smc(vs, N, ε, p; rng=rng)
        ε_vec = nothing
    end

    println("Total log-likelihood : $(round(sum(log_liks), digits=2))")
    println("Mean ESS             : $(round(mean(ess_vec), digits=1)) / $N")
    println("Min  ESS             : $(round(minimum(ess_vec), digits=1)) / $N")

    return (t=collect(1:T), ess=ess_vec, cum_ll=cumsum(log_liks),
            inc_ll=log_liks, eps_vec=ε_vec)
end

"""
    plot_smc_runs_faceted(runs)

Take a vector of named tuples `(label, method, result)` — where `result`
comes from `run_smc_result` — and produce faceted ggplot figures via RCall:
facets by `label` (e.g. different η values), coloured by `method` (e.g.
fixed vs. adaptive ε). Produces ESS, cumulative log-lik, and incremental
log-lik plots for all runs, plus an ε* plot restricted to adaptive runs.
"""
function plot_smc_runs_faceted(runs)
    eta_v = String[]; method_v = String[]; t_v = Int[]
    ess_v = Float64[]; cumll_v = Float64[]; incll_v = Float64[]

    eta_eps = String[]; method_eps = String[]; t_eps = Int[]; eps_v = Float64[]

    for run in runs
        res = run.result
        T = length(res.t)
        for i in 1:T
            push!(eta_v, run.label);  push!(method_v, run.method)
            push!(t_v, res.t[i])
            push!(ess_v, res.ess[i]); push!(cumll_v, res.cum_ll[i]); push!(incll_v, res.inc_ll[i])
        end
        if !isnothing(res.eps_vec)
            for i in 1:T
                push!(eta_eps, run.label); push!(method_eps, run.method)
                push!(t_eps, res.t[i]);    push!(eps_v, res.eps_vec[i])
            end
        end
    end

    has_eps = length(eps_v) > 0

    @rput eta_v method_v t_v ess_v cumll_v incll_v
    @rput eta_eps method_eps t_eps eps_v

    R"""
    library(ggplot2)

    df <- data.frame(eta = eta_v, method = method_v, t = t_v,
                      ess = ess_v, cum_ll = cumll_v, inc_ll = incll_v)

    p_ess <- ggplot(df, aes(x = t, y = ess, colour = method)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ eta) +
      labs(x = "t", y = "ESS", colour = "Method", title = "ESS over time") +
      theme_bw(base_size = 12) +
      theme(strip.background = element_blank())
    ggsave("figs/smc_run_ess.pdf", p_ess, width = 9, height = 4)

    p_cum <- ggplot(df, aes(x = t, y = cum_ll, colour = method)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ eta) +
      labs(x = "t", y = "cumulative log-lik", colour = "Method",
           title = "Cumulative log p(v[1:t])") +
      theme_bw(base_size = 12) +
      theme(strip.background = element_blank())
    ggsave("figs/smc_run_cumloglik.pdf", p_cum, width = 9, height = 4)

    p_inc <- ggplot(df, aes(x = t, y = inc_ll, colour = method)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ eta) +
      labs(x = "t", y = "log-lik increment", colour = "Method",
           title = "Incremental log p(v[t] | v[1:t-1])") +
      theme_bw(base_size = 12) +
      theme(strip.background = element_blank())
    ggsave("figs/smc_run_incloglik.pdf", p_inc, width = 9, height = 4)

    if (length(eps_v) > 0) {
      df_eps <- data.frame(eta = eta_eps, method = method_eps, t = t_eps, eps_star = eps_v)
      p_eps <- ggplot(df_eps, aes(x = t, y = eps_star, colour = method)) +
        geom_line(linewidth = 0.8) +
        facet_wrap(~ eta) +
        scale_y_log10() +
        labs(x = "t", y = expression(epsilon^"*"), colour = "Method",
             title = "Adaptive eps* over time") +
        theme_bw(base_size = 12) +
        theme(strip.background = element_blank())
      ggsave("figs/smc_run_epsstar.pdf", p_eps, width = 9, height = 4)
    }
    """

    msg = "Saved faceted plots: figs/smc_run_ess.pdf, figs/smc_run_cumloglik.pdf, " *
          "figs/smc_run_incloglik.pdf"
    msg *= has_eps ? ", figs/smc_run_epsstar.pdf" : ""
    println(msg)
end

# Example calls: fixed vs. adaptive ε, faceted by η
runs = [
    (label="η=1.0", method="fixed (ε=0)", result=run_smc_result(Para(η=1.0); adaptive=false, ε=0.0)),
    (label="η=1.0", method="adaptive",    result=run_smc_result(Para(η=1.0); adaptive=true)),
    (label="η=3.0", method="fixed (ε=0)", result=run_smc_result(Para(η=3.0); adaptive=false, ε=0.0)),
    (label="η=3.0", method="adaptive",    result=run_smc_result(Para(η=3.0); adaptive=true)),
]
plot_smc_runs_faceted(runs)