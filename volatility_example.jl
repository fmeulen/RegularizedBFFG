wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack

include("bffg.jl")

### Parameters
p = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ=√((π^2)/2))
# with larger variance
mult_var = 3.0
p_mv = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ = √(mult_var*(π^2)/2))

Random.seed!(10)
x0 = samplefromstationary(p) # Sample x0 from the stationary distribution 
S = 500 # Correct for time 0 (called tot_steps)  (Time steps)

Z = randn(S)
X = forward(x0, S, p, Z)
R = exp.(X/2) .* randn(S)
V = log.(R.^2)
#U = V - X

# plotting
p1 = plot(X, label="X", title="latent")
p2 = plot(V, label="V", title="observed")
plot(p1, p2)


########################################################

bf = backwardfilter(V, p)
bf_mv = backwardfilter(V, p_mv)

Zᵒ = randn(S)
Xᵒ = forwardguide(x0, bf, p, Zᵒ)
Xᵒmv = forwardguide(x0, bf, p_mv, Zᵒ)

plot!(p1, Xᵒ, color="red", label="Xᵒ")
plot!(p1, Xᵒmv, color="green", label="Xᵒmv")