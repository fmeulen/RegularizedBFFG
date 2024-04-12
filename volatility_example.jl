wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack
using SpecialFunctions

include("bffg.jl")

# forward model
# observe Vᵢ = Xᵢ + Uᵢ', where 
# Uᵢ' ~ log Uᵢ², where Uᵢ ~ N(0,1)
# Xᵢ = ω + ψ X_{i-1} + η Wᵢ, where Wᵢ ~ N(0,1)

### Parameters
p = (ω =  0.0, ψ = 0.9, η=0.163, μ=-1.27, σ=√((π^2)/2))
# with larger variance
mult_var = 3.0
p_mv = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ = √(mult_var*(π^2)/2))

#Random.seed!(10)
x0 = samplefromstationary(p) # Sample x0 from the stationary distribution 
S = 500 # Correct for time 0 (called tot_steps)  (Time steps)

Z = randn(S)
X = forward(x0, S, p, Z)
R = exp.(X/2) .* randn(S)
V = log.(R.^2)

# plotting
p1 = plot(X, label="X", title="latent")
p2 = plot(V, label="V", title="observed")
plot(p1, p2)


########################################################

# assess effect of ϵ

bf = backwardfilter(V, p)

Zᵒ = randn(S)
ϵ = 0.002

Xᵒ, λs, lw, guids = forwardguide(x0, bf, p, Zᵒ,V, ϵ)
Xᵒ0, λs0, lw0, guids0 = forwardguide(x0, bf, p, Zᵒ,V, 0.0)

pX = plot(X,legend = :outertop, label="X")
plot!(pX, Xᵒ, color="red", label="Xᵒ")
plot!(pX, Xᵒ0, color="green", label="Xᵒ0")

plw = plot(lw0, color="green", label="logweigth, ϵ=0",legend = :outertop)
plw = plot!(plw, lw, color="red", label="logweigth")

plλ = plot(guids, color="blue", label="guids")
plot!(plλ, λs, color="red", label="λ",legend = :outertop)

l = @layout [a;b;c;d]
pall = plot(pX, plw, plλ, layout=l,size = (600, 1000))

savefig(pX, "paths.png")
savefig(plw, "weights.png")
savefig(plλ, "lambdas.png")
savefig(pall, "all.png")

# sum of logweights is decreasing in ϵ:
Zᵒ = randn(S)
ϵs = 0.0:0.01:10.0
sumlw = sumlogweights(x0, bf, p, Zᵒ, V).(ϵs) 
plot(ϵs, sumlw)

@show std(exp.(lw)), smc_ess(exp.(lw))
@show std(exp.(lw0)), smc_ess(exp.(lw0))