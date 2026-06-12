using DifferentialEquations
using Statistics
using LinearAlgebra
using Random
using Plots
using DataFrames

Random.seed!(42)

"""
    setup_sde_model()
    
    Uses STABLE parameters (ψ < 0 for mean-reverting OU process)
    """
function setup_sde_model()
    ω = 0.5        # Drift intercept  
    ψ = -0.9       # Drift slope (negative for MEAN-REVERTING/stable)
    q = 1.0        # Diffusion coefficient (reduced for stability)
    
    T_final = 1.0  # Shorter horizon to limit error accumulation
    dt = 0.01
    
    v_obs = 1.0    # Observed value
    σ_obs = 0.5    # Observation noise
    
    return (; ω, ψ, q, T_final, dt, v_obs, σ_obs)
end

function get_guide_g(t, x, params)
    @unpack ω, ψ, q, T_final, v_obs, σ_obs = params
    τ = max(T_final - t, 1e-9)
    
    M = x * exp(ψ * τ) + (ω / ψ) * (exp(ψ * τ) - 1)
    S_sq = (q^2 / (2 * abs(ψ))) * (1 - exp(2 * ψ * τ))
    
    total_var = S_sq + σ_obs^2
    
    # Compute in log-space for stability
    log_g = -0.5 * ((v_obs - M)^2 / total_var) - 0.5 * log(2π * total_var)
    return exp(log_g)
end

function score_grad_log_g(t, x, params)
    @unpack ω, ψ, q, T_final, v_obs, σ_obs = params
    τ = max(T_final - t, 1e-9)
    
    dM_dx = exp(ψ * τ)
    S_sq = (q^2 / (2 * abs(ψ))) * (1 - exp(2 * ψ * τ))
    total_var = S_sq + σ_obs^2
    
    M = x * dM_dx + (ω / ψ) * (exp(ψ * τ) - 1)
    
    grad_log_g = (v_obs - M) * dM_dx / total_var
    return grad_log_g
end

function sde_drift_guided(t, x, p::NamedTuple{(:params, :epsilon)})
    (; params, epsilon) = p
    @unpack ω, ψ, q = params
    
    b_orig = ω + ψ * x
    g_val = get_guide_g(t, x, params)
    grad_log_g = score_grad_log_g(t, x, params)
    
    denom = g_val + epsilon
    if denom < 1e-16
        r_eps = 0.0
    else
        r_eps = (g_val * grad_log_g) / denom
    end
    
    # CLIP the drift modification to prevent explosions
    r_eps = clamp(r_eps, -10.0, 10.0)
    
    return b_orig + (q^2) * r_eps
end

function sde_drift_original(t, x, p::NamedTuple{(:params,)})
    @unpack ω, ψ = p.params
    return ω + ψ * x
end

function calculate_m_epsilon_debug(params, ε; N_mc=500, dt=0.01)
    @unpack ω, ψ, q, T_final, v_obs, σ_obs = params
    
    tspan = (0.0, T_final)
    ts = collect(0:dt:T_final)
    x0 = 0.0
    
    log_m_est = 0.0
    count_valid = 0
    count_overflow = 0
    count_nan = 0
    
    println("  Starting $N_mc simulations with ε=$ε...")
    
    for i in 1:N_mc
        try
            prob = SDEProblem((u,p,t)->sde_drift_guided(t,u,p), (u,p,t)->q, 
                              tspan, x0, (params=params, epsilon=ε))
            
            sol = solve(prob, EulerMaruyama(), dt=dt, saveat=ts, adaptive=false)
            path = sol.u
            
            if any(isnan.(path)) || any(isinf.(path))
                count_nan += 1
                continue
            end
            
            h_val = pdf(Normal(path[end], σ_obs), v_obs)
            
            # LOG-SPACE Girsanov calculation
            log_RN = 0.0
            current_x = x0
            
            for k in 1:length(ts)-1
                t_k = ts[k]
                dt_k = dt
                x_curr = path[k]
                x_next = path[k+1]
                
                b_mod_k = sde_drift_guided(t_k, x_curr, (params=params, epsilon=ε))
                b_orig_k = sde_drift_original(t_k, x_curr, (params=params,))
                
                diff_drift = b_mod_k - b_orig_k
                
                dW = (x_next - x_curr - b_mod_k * dt_k) / q
                
                factor = diff_drift / q
                # Clip to prevent overflow
                term1 = -factor * dW
                term2 = -0.5 * factor^2 * dt_k
                term1 = clamp(term1, -100.0, 100.0)
                term2 = clamp(term2, -100.0, 0.0)
                
                log_RN += term1 + term2
                
                if !isfinite(log_RN)
                    count_overflow += 1
                    break
                end
            end
            
            if !isfinite(log_RN)
                continue
            end
            
            log_h = log(max(h_val, 1e-300))
            log_G = log_h + log_RN
            
            # Clip log_G before exponentiating
            log_G = clamp(log_G, -100.0, 100.0)
            
            G_eps = exp(log_G)
            
            if isfinite(G_eps) && G_eps < 1e150
                log_m_est += 2 * log_G  # Accumulate log of squared values
                count_valid += 1
            else
                count_overflow += 1
            end
            
        catch e
            count_nan += 1
            continue
        end
        
        if i % 100 == 0
            println("    Completed $i/$N_mc runs (valid: $count_valid)")
        end
    end
    
    println("  Results: valid=$count_valid, overflow=$count_overflow, nan=$count_nan")
    
    if count_valid == 0
        println("  ⚠️  NO VALID SAMPLES - m(ε) = Inf")
        return Inf, 0
    end
    
    # Compute mean of log-squared values and convert back
    mean_log_G_squared = log_m_est / count_valid
    m_approx = exp(mean_log_G_squared)
    
    println("  → Log mean of G² = $(mean_log_G_squared)")
    println("  → m(ε) ≈ $(round(m_approx, digits=4))")
    
    return m_approx, count_valid
end

# --- Main Execution ---

println("="^50)
println("Debug: Approximating m(ε) for Guided SDE")
println("="^50)

params = setup_sde_model()
println("\nParameters:")
println("  ω=$(params.ω), ψ=$(params.ψ), q=$(params.q)")
println("  T_final=$(params.T_final), σ_obs=$(params.σ_obs)")
println("")

epsilons = [0.0, 0.001, 0.01, 0.1, 1.0]
results = []
valid_counts = []

for ε in epsilons
    val, cnt = calculate_m_epsilon_debug(params, ε, N_mc=200)
    push!(results, (ε=ε, m_approx=val))
    push!(valid_counts, cnt)
end

println("\n" * "="^50)
println("Summary:")
for (r, c) in zip(results, valid_counts)
    status = isfinite(r.m_approx) ? "✓" : "✗"
    println("  $status ε=$(r.ε): m≈$(r.m_approx) (valid samples: $c)")
end