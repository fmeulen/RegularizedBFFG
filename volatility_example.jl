wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack
using SpecialFunctions

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

ϵ = 0.01

Zᵒ = randn(S)
Xᵒ, λs, lw = forwardguide(x0, bf, p, Zᵒ,V, ϵ)
Xᵒmv, λs_mv, lw_mv = forwardguide(x0, bf, p_mv, Zᵒ,V, ϵ)


lw = logweights(x0, Xᵒ, V, p, bf, ϵ)
lw0 = logweights(x0, Xᵒ, V, p, bf, 0.0)
#lw_mv = logweights(x0, Xᵒmv, V, p_mv, bf_mv; ϵ)


plot!(p1, Xᵒ, color="red", label="Xᵒ")
plot!(p1, Xᵒmv, color="green", label="Xᵒmv")

plot(λs)
plot!(λs_mv, col="red")

plot(lw)
plot!(lw0)


# Monte Carlo study
bf = backwardfilter(V, p)

Zᵒ = randn(S)
ϵ = 0.02
Xᵒ, λs, lw = forwardguide(x0, bf, p, Zᵒ,V, ϵ)
Xᵒ0, λs0, lw0 = forwardguide(x0, bf, p, Zᵒ,V, 0.0)

pX = plot(X)
plot!(pX, Xᵒ, color="red", label="Xᵒ")
plot!(pX, Xᵒ0, color="green", label="Xᵒ0")

plw = plot(lw0, color="green", label="logweigth, ϵ=0")
plw = plot!(plw, lw, color="red", label="logweigth")


plλ = plot(λs, color="red", label="λ")
plot!(plλ, λs0, color="green", label="λ, ϵ=0")

plot(pX, plw, plλ)

Zᵒ = randn(S)
ϵs = 0.0:0.01:10.0
sumlw = sumlogweights(x0, bf, p, Zᵒ, V).(ϵs) 
plot(ϵs, sumlw)

savefig(pX, "paths.png")
savefig(plw, "weights.png")
savefig(plλ, "lambdas.png")