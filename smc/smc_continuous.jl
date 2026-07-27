wd = @__DIR__
cd(wd)

using Distributions
using Random
using Statistics
using StatsBase
using Plots
using Parameters
using UnPack

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
    return log(hT) + log(g0 + ε) - log(gT + ε) - log_psi
end

# ── m̂(ε) and its derivative ───────────────────────────────────────────────────

function mhat_and_deriv(ε, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
    m  = 0.0
    dm = 0.0
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
        Ψ_inv = exp(-lp)
        if !isfinite(Ψ_inv)
            continue
        end

        # ∂_ε log Ψ = -Σ_i db_score_i · g_i / (g_i+ε)²
        dlp = 0.0
        @inbounds for i in eachindex(gvals)
            dlp -= db_score[i] * gvals[i] / (gvals[i] + ε)^2
        end

        denom = gT + ε
        g0ε   = g0 + ε
        core  = h2 / denom * Ψ_inv

        # m̂ contribution
        contrib = g0ε * core
        if isfinite(contrib)
            m += Ws[n] * contrib
        end

        # dm̂/dε contribution
        # = core·(1 - g0ε/denom - g0ε·dlp)
        dcontrib = core * (1.0 - g0ε/denom - g0ε*dlp)
        if isfinite(dcontrib)
            dm += Ws[n] * dcontrib
        end
    end
    return m, dm
end

# ── Brent's method for zero of dm̂/dε ─────────────────────────────────────────

function brent_eps_star(precomp, x0s, hTs, gTs, Ws, v, T_obs, p;
                        ε_lo=1e-10, ε_hi=1.0, tol=1e-8, maxiter=100)
    _, d_lo = mhat_and_deriv(ε_lo, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
    _, d_hi = mhat_and_deriv(ε_hi, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)

    # no sign change → return boundary with smaller m̂
    if d_lo * d_hi > 0
        m_lo, _ = mhat_and_deriv(ε_lo, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
        m_hi, _ = mhat_and_deriv(ε_hi, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
        return m_lo < m_hi ? ε_lo : ε_hi
    end

    # Brent's method on dm̂/dε = 0
    a, b_  = ε_lo, ε_hi
    fa, fb = d_lo, d_hi
    c, fc  = a, fa
    d_b    = b_ - a
    e_b    = d_b

    for _ in 1:maxiter
        if fb * fc > 0
            c, fc = a, fa
            d_b   = b_ - a
            e_b   = d_b
        end
        if abs(fc) < abs(fb)
            a, b_, c   = b_, c, b_
            fa, fb, fc = fb, fc, fb
        end
        tol1 = 2eps(Float64)*abs(b_) + 0.5*tol
        xm   = 0.5*(c - b_)
        if abs(xm) <= tol1 || fb == 0
            return b_
        end
        if abs(e_b) >= tol1 && abs(fa) > abs(fb)
            s = fb / fa
            if a ≈ c
                p_b = 2*xm*s
                q_b = 1 - s
            else
                q_b = fa/fc
                r_b = fb/fc
                p_b = s*(2*xm*q_b*(q_b - r_b) - (b_ - a)*(r_b - 1))
                q_b = (q_b - 1)*(r_b - 1)*(s - 1)
            end
            if p_b > 0
                q_b = -q_b
            else
                p_b = -p_b
            end
            if 2*p_b < min(3*xm*q_b - abs(tol1*q_b), abs(e_b*q_b))
                e_b = d_b
                d_b = p_b/q_b
            else
                d_b = xm
                e_b = d_b
            end
        else
            d_b = xm
            e_b = d_b
        end
        a, fa = b_, fb
        b_ += abs(d_b) > tol1 ? d_b : (xm > 0 ? tol1 : -tol1)
        _, fb = mhat_and_deriv(b_, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
    end
    return b_
end

# ── Find ε* by both grid search and Brent's method ───────────────────────────

function find_eps_star(precomp, x0s, hTs, gTs, Ws, v, T_obs, p, ε_grid)
    K    = length(ε_grid)
    mhat = zeros(K)

    for (k, ε) in enumerate(ε_grid)
        m, _ = mhat_and_deriv(ε, precomp, x0s, hTs, gTs, Ws, v, T_obs, p)
        mhat[k] = m
    end

    ε_star_grid  = ε_grid[argmin(mhat)]
    ε_star_brent = brent_eps_star(precomp, x0s, hTs, gTs, Ws, v, T_obs, p;
                                   ε_lo=ε_grid[1], ε_hi=ε_grid[end])

    return ε_star_grid, ε_star_brent, mhat
end

# ── Adaptive particle filter ──────────────────────────────────────────────────

function adaptive_pf(vs, T_seg, p;
                     N        = 500,
                     seed     = 42,
                     ε_grid   = exp.(range(log(1e-8), log(1.0), length=50)),
                     ε_fix    = nothing,   # fixed ε; nothing → adaptive
                     use_brent = true)     # true → Brent, false → grid

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
        if isnothing(ε_fix)
            ε_star_grid, ε_star_brent, _ = find_eps_star(
                precomp, xs, hTs_aux, gTs_aux, Ws, v, T_seg, p, ε_grid)
            ε_star = use_brent ? ε_star_brent : ε_star_grid
        else
            ε_star = ε_fix
        end
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

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    T_seg = 1.0
    ψ     = 0.9
    n_obs = 50
    N     = 500
    R     = 100

    configs = [
        (η=1.0, α=0.0, label="eta1_lin"),
        (η=3.0, α=0.0, label="eta3_lin"),
        (η=3.0, α=0.5, label="eta3_nln"),
    ]

    ε_grid = exp.(range(log(1e-8), log(1.0), length=50))

    for cfg in configs
        println("\n=== $(cfg.label) ===")
        p  = ParaCT(ψ, cfg.η, cfg.α, T_seg)
        vs = simulate_data(n_obs, T_seg, p; seed=1)

        results = Dict(
            "adaptive (grid)"  => (ε_fix=nothing, use_brent=false),
            "adaptive (Brent)" => (ε_fix=nothing, use_brent=true),
            "fixed ε=0"        => (ε_fix=0.0,     use_brent=false),
        )

        minESS_all = Dict(k => Float64[] for k in keys(results))
        ll_all     = Dict(k => Float64[] for k in keys(results))

        for r in 1:R
            r % 10 == 0 && print("  run $r/$R\r")
            for (label, opts) in results
                ll, ess, _ = adaptive_pf(vs, T_seg, p; N=N, seed=r,
                                          ε_grid=ε_grid,
                                          ε_fix=opts.ε_fix,
                                          use_brent=opts.use_brent)
                push!(ll_all[label], ll)
                push!(minESS_all[label], minimum(ess))
            end
        end

        println()
        for (label, _) in results
            println("  $label: mean ll=$(round(mean(ll_all[label]),digits=1)), " *
                    "mean minESS=$(round(mean(minESS_all[label]),digits=1))")
        end

        colors = ["adaptive (grid)" => :blue,
                  "adaptive (Brent)" => :green,
                  "fixed ε=0" => :red]
        plt = plot(xlabel="min ESS", ylabel="P(min ESS ≤ x)",
                   title="min ESS CDF — $(cfg.label) (N=$N, R=$R)")
        for (label, col) in colors
            vals = sort(minESS_all[label])
            plot!(plt, vals, (1:R)./R, label=label, lw=2, color=col)
        end
        savefig(plt, "figs/ct_pf_$(cfg.label).png")
        println("  Saved figs/ct_pf_$(cfg.label).png")
    end
end

main()



# for grid optimisation, benchmark whether grid search or Brent is faster

using BenchmarkTools
ψ = 0.9
p  = ParaCT(ψ, 3.0, 0.0, 1.0)
vs = simulate_data(50, 1.0, p; seed=1)
v  = vs[1]

rng     = MersenneTwister(42)
xs      = fill(p.x0, 500)
xs_aux  = [simulate_unguided_segment(xs[n], 1.0, p, rng) for n in 1:500]
xTs_aux = [xs_aux[n][end] for n in 1:500]
hTs_aux = [h_ct(xTs_aux[n], v) for n in 1:500]
gTs_aux = [g_ct(1.0, xTs_aux[n], v, 1.0, p) for n in 1:500]
precomp = [precompute_path(xs_aux[n], v, 1.0, p) for n in 1:500]
Ws      = fill(1.0/500, 500)
ε_grid  = exp.(range(log(1e-8), log(1.0), length=50))


# grid search only
@btime begin
    K    = length($ε_grid)
    mhat = zeros(K)
    for (k, ε) in enumerate($ε_grid)
        m, _ = mhat_and_deriv(ε, $precomp, $xs, $hTs_aux, $gTs_aux, $Ws, $v, 1.0, $p)
        mhat[k] = m
    end
    $ε_grid[argmin(mhat)]
end

# Brent only
@btime brent_eps_star($precomp, $xs, $hTs_aux, $gTs_aux, $Ws, $v, 1.0, $p;
                       ε_lo=$ε_grid[1], ε_hi=$ε_grid[end])

# so Brent is about 3 times faster and moreover more accurate