wd = @__DIR__
cd(wd)

using Distributions
using Random
using Statistics
using StatsBase
using RCall
using Parameters
using UnPack
using Optim

mkpath("figs")

# ── Parameters ────────────────────────────────────────────────────────────────

@with_kw struct ParaCT{R}
    x0::R        = 0.0
    κ::R         = 1.0
    α::R         = 0.0
    q::R         = 1.0
    μ_χ::R       = -1.27
    σ_χ²::R      = π^2/2
    n_steps::Int = 200
end

function ParaCT(ψ, η, α, T_seg)
    κ0 = -log(ψ) / T_seg
    q0 = sqrt(2κ0 * η^2 / (1 - ψ^2))
    ParaCT(κ=κ0, q=q0, α=α)
end

# ── Drifts ────────────────────────────────────────────────────────────────────

b(x, p)     = -p.κ * x + p.α * sin(x)
b_ref(x, p) = -p.κ * x

# ── Observation density ───────────────────────────────────────────────────────

function h_ct(xT, v)
    u = exp(v - xT)
    return pdf(Chisq(1), u) * u
end

# ── g(t, x) via Gaussian convolution ─────────────────────────────────────────

function g_ct(t, x, v, T_obs, p)
    @unpack κ, q, μ_χ, σ_χ² = p
    τ   = T_obs - t
    η²τ = q^2/(2κ) * (1 - exp(-2κ*τ))
    return pdf(Normal(v - μ_χ, sqrt(σ_χ² + η²τ)), exp(-κ*τ) * x)
end

function score_g(t, x, v, T_obs, p)
    @unpack κ, q, μ_χ, σ_χ² = p
    τ   = T_obs - t
    η²τ = q^2/(2κ) * (1 - exp(-2κ*τ))
    ψτ  = exp(-κ*τ)
    Σ   = σ_χ² + η²τ
    return -ψτ * (ψτ*x - (v - μ_χ)) / Σ
end

function r_eps(t, x, ε, v, T_obs, p)
    gval = g_ct(t, x, v, T_obs, p)
    return (gval / (gval + ε)) * score_g(t, x, v, T_obs, p)
end

# ── Precompute path quantities for fast ε-loop ────────────────────────────────

function precompute_path(xs, v, T_obs, p)
    @unpack n_steps = p
    dt       = T_obs / n_steps
    ts       = range(0.0, T_obs, length=n_steps+1)
    db_score = Vector{Float64}(undef, n_steps)
    gvals    = Vector{Float64}(undef, n_steps)
    for i in 1:n_steps
        x            = xs[i]
        gvals[i]     = g_ct(ts[i], x, v, T_obs, p)
        db_score[i]  = (b(x, p) - b_ref(x, p)) * score_g(ts[i], x, v, T_obs, p) * dt
    end
    return db_score, gvals
end

function log_psi_fast(db_score, gvals, ε)
    log_psi = 0.0
    @inbounds for i in eachindex(gvals)
        log_psi += db_score[i] * gvals[i] / (gvals[i] + ε)
    end
    return log_psi
end

# ── Simulate path segments ────────────────────────────────────────────────────

function simulate_unguided_segment(x0, T_obs, p, rng)
    @unpack q, n_steps = p
    dt   = T_obs / n_steps
    sqdt = sqrt(dt)
    xs   = Vector{Float64}(undef, n_steps + 1)
    xs[1] = x0
    for i in 1:n_steps
        dW      = randn(rng) * sqdt
        xs[i+1] = xs[i] + b(xs[i], p) * dt + q * dW
    end
    return xs
end

function simulate_guided_segment(x0, ε, v, T_obs, p, rng)
    @unpack q, n_steps = p
    dt   = T_obs / n_steps
    sqdt = sqrt(dt)
    Q    = q^2
    xs      = Vector{Float64}(undef, n_steps + 1)
    xs[1]   = x0
    log_psi = 0.0
    ts      = range(0.0, T_obs, length=n_steps+1)

    for i in 1:n_steps
        t  = ts[i]
        x  = xs[i]
        dW = randn(rng) * sqdt

        rval      = r_eps(t, x, ε, v, T_obs, p)
        db        = b(x, p) - b_ref(x, p)
        log_psi  += db * rval * dt
        xs[i+1]   = x + (b(x, p) + Q*rval)*dt + q*dW

        if !isfinite(xs[i+1]) || abs(xs[i+1]) > 1e4
            return x0, -1e6
        end
    end
    return xs[end], log_psi
end

# ── IS weight ─────────────────────────────────────────────────────────────────

function log_weight_segment(xT, log_psi, x0, ε, v, T_obs, p)
    g0 = g_ct(0.0, x0, v, T_obs, p)
    gT = g_ct(T_obs, xT, v, T_obs, p)
    hT = h_ct(xT, v)
    if hT <= 0.0 || gT + ε <= 0.0 || g0 + ε <= 0.0
        return -Inf
    end
    return log(hT) + log(g0 + ε) - log(gT + ε) + log_psi
end

# ── m̂(ε) ─────────────────────────────────────────────────────────────────────

function mhat_eps(ε, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
    m = 0.0
    for n in eachindex(precomp)
        hT = hTs[n]
        gT = gTs[n]
        h2 = hT^2
        if h2 == 0.0 || gT + ε <= 0.0
            continue
        end
        db_score, gvals = precomp[n]
        g0  = g_ct(0.0, x0s[n], v, T_obs, p)
        lp  = log_psi_fast(db_score, gvals, ε)
        if isnan(lp)
            continue
        end
        Ψ = exp(lp)
        # if !isfinite(Ψ_inv)
        #     continue
        # end

        denom   = gT + ε
        g0ε     = g0 + ε
        contrib = g0ε * h2 / denom * Ψ
        if isfinite(contrib)
            m += Ws[n] * contrib
        end
    end
    return m
end

# ── Find ε* by univariate optimisation (Optim.jl, Brent's method) ────────────

function find_eps_star(precomp, x0s, hTs, gTs, Ws, v, T_obs, p;
                        ε_lo=1e-8, ε_hi=1.0)
    obj = ε -> mhat_eps(ε, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
    res = optimize(obj, ε_lo, ε_hi, Brent())
    return Optim.minimizer(res)
end

# ── Adaptive particle filter ──────────────────────────────────────────────────

function adaptive_pf(vs, T_seg, p;
                     N     = 500,
                     seed  = 42,
                     ε_lo  = 1e-8,
                     ε_hi  = 1.0,
                     ε_fix = nothing)   # fixed ε; nothing → adaptive (Optim)

    rng    = MersenneTwister(seed)
    n_obs  = length(vs)

    xs     = fill(p.x0, N)
    log_ws = zeros(N)
    log_ml = 0.0
    ess_t  = Float64[]
    eps_t  = Float64[]

    for i in 1:n_obs
        v = vs[i]

        # normalised carry-over weights
        lw_max = maximum(log_ws)
        ws     = exp.(log_ws .- lw_max)
        Ws     = ws ./ sum(ws)
        ess    = 1.0 / sum(Ws.^2)

        # resample if ESS < N/2, following eq (10.3)
        resampled = false
        if ess < N/2
            log_ml   += lw_max + log(mean(ws))
            idx       = sample(rng, 1:N, Weights(Ws), N)
            xs        = xs[idx]
            log_ws    = zeros(N)
            Ws        = fill(1.0/N, N)
            resampled = true
        end

        # simulate unguided auxiliary paths and precompute
        xs_aux  = [simulate_unguided_segment(xs[n], T_seg, p, rng) for n in 1:N]
        xTs_aux = [xs_aux[n][end] for n in 1:N]
        hTs_aux = [h_ct(xTs_aux[n], v) for n in 1:N]
        gTs_aux = [g_ct(T_seg, xTs_aux[n], v, T_seg, p) for n in 1:N]
        precomp = [precompute_path(xs_aux[n], v, T_seg, p) for n in 1:N]

        # find ε*
        ε_star = isnothing(ε_fix) ?
            find_eps_star(precomp, xs, hTs_aux, gTs_aux, Ws, v, T_seg, p;
                          ε_lo=ε_lo, ε_hi=ε_hi) :
            ε_fix
        push!(eps_t, ε_star)

        # propose guided segments and update weights
        new_xs     = Vector{Float64}(undef, N)
        new_log_ws = Vector{Float64}(undef, N)
        for n in 1:N
            xT, lp        = simulate_guided_segment(xs[n], ε_star, v, T_seg, p, rng)
            lw             = log_weight_segment(xT, lp, xs[n], ε_star, v, T_seg, p)
            new_xs[n]     = xT
            new_log_ws[n] = log_ws[n] + (isfinite(lw) ? lw : -Inf)
        end
        xs     = new_xs
        log_ws = new_log_ws

        # likelihood increment following eq (10.3)
        lw_max = maximum(log_ws)
        ws     = exp.(log_ws .- lw_max)
        if !resampled
            log_ml += lw_max + log(mean(ws))
        end
        log_ws = log.(ws ./ sum(ws))

        Ws_new = exp.(log_ws)
        push!(ess_t, 1.0 / sum(Ws_new.^2))
    end

    return log_ml, ess_t, eps_t
end

# ── Simulate data ─────────────────────────────────────────────────────────────

function simulate_data(n_obs, T_seg, p; seed=1)
    rng = MersenneTwister(seed)
    x   = p.x0
    vs  = Float64[]
    for _ in 1:n_obs
        dt = T_seg / p.n_steps
        for _ in 1:p.n_steps
            x += b(x, p)*dt + p.q*randn(rng)*sqrt(dt)
        end
        Z = randn(rng)
        push!(vs, x + log(Z^2/2))
    end
    return vs
end


# ── Generic experiment runner: takes configs of (p, label, title) ────────────

function run_configs(configs; N=500, R=10, T_seg=1.0, n_obs=50)
    all_labels  = String[]
    all_methods = String[]
    all_minESS  = Float64[]

    for cfg in configs
        println("\n=== $(cfg.label) ===")
        p = cfg.p
        @show p
        vs = simulate_data(n_obs, T_seg, p; seed=1)

        results = Dict(
            "adaptive"   => (ε_fix=nothing,),
            "fixed ε=0"  => (ε_fix=0.0,),
        )

        minESS_all = Dict(k => Float64[] for k in keys(results))
        ll_all     = Dict(k => Float64[] for k in keys(results))

        for r in 1:R
            r % 10 == 0 && print("  run $r/$R\r")
            for (label, opts) in results
                ll, ess, _ = adaptive_pf(vs, T_seg, p; N=N, seed=r,
                                          ε_fix=opts.ε_fix)
                push!(ll_all[label], ll)
                push!(minESS_all[label], minimum(ess))
            end
        end

        println()
        for (label, _) in results
            println("  $label: mean ll=$(round(mean(ll_all[label]),digits=1)), " *
                    "mean minESS=$(round(mean(minESS_all[label]),digits=1))")
        end

        for (label, vals) in minESS_all
            append!(all_labels,  fill(cfg.label, length(vals)))
            append!(all_methods, fill(label, length(vals)))
            append!(all_minESS,  vals)
        end
    end

    facet_levels = [cfg.label for cfg in configs]
    facet_titles = [cfg.title for cfg in configs]

    return all_labels, all_methods, all_minESS, facet_levels, facet_titles
end

# ── Basic (native) experiment: fixed q, 2×2 grid of (κ, α) ───────────────────

function main_qκα(q, κ_vals::AbstractVector, α_vals::AbstractVector;
                   R=10, T_seg=1.0, n_obs=50, N=500)
    length(κ_vals) == 2 || throw(ArgumentError("κ_vals must have length 2"))
    length(α_vals) == 2 || throw(ArgumentError("α_vals must have length 2"))

    configs = vec([(p     = ParaCT(κ=κ, q=q, α=α),
                     label = "kappa$(i)_alpha$(j)",
                     title = "alpha == $(α) * ',' ~ kappa == $(κ)")
                    for (i, κ) in enumerate(κ_vals), (j, α) in enumerate(α_vals)])

    return run_configs(configs; N=N, R=R, T_seg=T_seg, n_obs=n_obs)
end

# ── Reparametrised version: a (ψ_ref, η_ref) reference pair pins q; κ_vals and
#    α_vals are still supplied directly (κ is NOT derived from ψ_ref) ────────

function main_ψη(ψ_ref, η_ref, κ_vals::AbstractVector, α_vals::AbstractVector;
                  R=10, T_seg=1.0, n_obs=50, N=500)
    κ_ref = -log(ψ_ref) / T_seg
    q     = sqrt(2κ_ref * η_ref^2 / (1 - ψ_ref^2))
    return main_qκα(q, κ_vals, α_vals; R=R, T_seg=T_seg, n_obs=n_obs, N=N)
end

function plotting(all_labels, all_methods, all_minESS, facet_levels, facet_titles)
    @rput all_labels all_methods all_minESS facet_levels facet_titles
    R"""
    library(ggplot2)
    df <- data.frame(label = factor(all_labels, levels = facet_levels),
                      method = all_methods,
                      minESS = all_minESS)
    label_map <- setNames(facet_titles, facet_levels)
    p <- ggplot(df, aes(x = minESS, color = method)) +
        stat_ecdf(linewidth = 1) +
        facet_wrap(~ label, nrow = 2, ncol = 2,
                   labeller = as_labeller(label_map, default = label_parsed)) +
        labs(x = "min ESS", y = "P(min ESS <= x)", color = "Method") +
        theme_bw() + theme(legend.position="bottom")
    ggsave("figs/ct_pf_facet.pdf", p, width = 9, height = 7, dpi = 150)
    """
    println("  Saved figs/ct_pf_facet.pdf")
end


labels, methods, minESS, levels, titles = main_qκα(.4, [0.5, 3.0], [0.0, 4.0]; R=100)
plotting(labels, methods, minESS, levels, titles)

labels, methods, minESS, levels, titles = main_qκα(2.8, [0.5, 3.0], [0.0, 4.0]; R=100)
plotting(labels, methods, minESS, levels, titles)


# or, reparametrized:
labels, methods, minESS, levels, titles = main_ψη(0.9, 1.0, [0.5, 3.0], [0.0, 1.5])
plotting(labels, methods, minESS, levels, titles)


